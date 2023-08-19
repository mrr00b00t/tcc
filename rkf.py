import os
import joblib
import numpy as np

from itertools import product
from multiprocessing.pool import ThreadPool

import numpy as np
from pmlb import fetch_data

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, pairwise_distances

from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.core.problem import ElementwiseProblem, StarmapParallelization

from configs import DSETS, CS, DEGREES, POPS_SIZE, NS_GEN, NS_SPLITS, TEST_SIZES, EV_CHECKS, IVT_CHECKS, SEEDS


class RKFP(ElementwiseProblem):
    def __init__(self, C, degree, dset, n_splits, test_size, ev_check, ivt_check, seed, **kwargs):
        assert isinstance(degree, int) and C > 0.0, 'degree must be integer and must be strictly greater than zero'
        assert isinstance(degree, int) and (3 <= degree <= 7), 'degree must be integer and must be in [3, 5]'
        assert isinstance(dset, str), 'dataset name must be str'
        assert isinstance(test_size, float) and (0.0 < test_size < 1.0), 'fraction of test set must be float and must be in (0.0, 1.0)'
        assert isinstance(n_splits, int) and (2 < n_splits < 6), 'number of splits must be integer and must be in [3, 5]'
        assert isinstance(seed, int) and (seed >= 2), 'seed must be integer and equal or greater than 2'
        assert isinstance(ev_check, bool), 'eigenvalue check must be boolean'
        assert isinstance(ivt_check, bool), 'intermediate value theorem check must be boolean'
        
        self.C = C
        self.degree = degree
        self.dset = dset
        self.n_splits = n_splits
        self.test_size = test_size
        self.ev_check = ev_check
        self.ivt_check = ivt_check
        self.seed = seed
        
        super().__init__(
            n_var=2*self.degree, n_obj=1, n_ieq_constr=0, n_eq_constr=2,
            xl=-1.0, xu=+1.0, elementwise_evaluation=True, **kwargs
        )
        
    def arr2poly(self, a):
        if self.degree == 3: return lambda x: (a[0]*x + a[1])*x + a[2]
        if self.degree == 4: return lambda x: ((a[0]*x + a[1])*x + a[2])*x + a[3]
        if self.degree == 5: return lambda x: (((a[0]*x + a[1])*x + a[2])*x + a[3])*x + a[4]
            
    def num_den(self, a):
        b = a.copy() / a[-1]
        num = self.arr2poly(b[:self.degree])
        den = self.arr2poly(b[self.degree:])
        return num, den
    
    @staticmethod
    def ev_constraint(G): return 1.0 if np.any(np.linalg.eigvalsh(G) < 0) else 0.0
    
    @staticmethod
    def ivt_constraint(low, high, den):
        y = den(np.random.uniform(low=low, high=high, size=10000))
        return 1.0 if np.any(y < 0) and np.any(y > 0) else 0.0
    
    def _evaluate(self, x, out, *args, **kwargs):
        num, den = self.num_den(x)
        f = lambda x: num(x) / den(x)
        
        dset_X, dset_y = fetch_data(dataset_name=self.dset, return_X_y=True, local_cache_dir='datasets')
        Xn,  _, yn,  _ = train_test_split(dset_X, dset_y, test_size=self.test_size, random_state=self.seed, shuffle=True)
        rs = RobustScaler()
        Xns = rs.fit_transform(Xn)
        
        max_pd, l_bas, l_itr, l_nsv = -1, list(), list(), list()
        
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=np.random.randint(2, 99999))
        for train_index, test_index in kf.split(Xns, yn):
            X0s, X1s, y0, y1 = Xns[train_index], Xns[test_index], yn[train_index], yn[test_index]
            
            pd00 = pairwise_distances(X0s, X0s)
            G0 = f(pd00) # train gram matrix
            pd10 = pairwise_distances(X1s, X0s)
            G1 = f(pd10) # valid gram matrix
            
            max_pd = max(max_pd, pd00.max(), pd10.max())
            
            svc = SVC(C=self.C, kernel='precomputed')
            svc.fit(G0, y0)
            yp = svc.predict(G1)
            
            l_bas.append( accuracy_score(y1, yp) )
            l_itr.append( np.mean(svc.n_iter_) )
            l_nsv.append( svc.support_.shape[0] / X0s.shape[0] )
            
        bas = np.mean(l_bas)
        itr = np.mean(l_itr)
        nsv = np.mean(l_nsv)
        
        H0 = 0.0
        if self.ev_check:
            Xes = np.random.choice(Xns.tolist(), size=5, replace=False)
            pdee = pairwise_distances(Xes, Xes)
            S = np.random.randn(5, 5) * 0.075 + 1.0
            S = np.tril(S) + np.tril(S, k=-1).T
            pdee = pdee * S
            pdee[pdee < 0] = 0
            Ge = f(pdee)
            max_pd = max(max_pd, pdee.max()) # max pd for ivt check
            H0 = RKFP.ev_constraint(G=Ge)
        
        H1 = 0.0
        if self.ivt_check:
            H1 = RKFP.ivt_constraint(low=0.0, high=max_pd, den=den)
        
        out['H'] = [H0, H1]
        
        alpha = 0.003619120682527099
        beta = 0.05
        out['F'] = 1 - bas + alpha * np.log(itr) + beta * nsv
        
    def _evaluate2(self, x):
        num, den = self.num_den(x)
        f = lambda x: num(x) / den(x)
        
        dset_X, dset_y = fetch_data(dataset_name=self.dset, return_X_y=True, local_cache_dir='datasets')
        X0, X1, y0, y1 = train_test_split(dset_X, dset_y, test_size=self.test_size, random_state=self.seed, shuffle=True)
        
        rs = RobustScaler()
        X0s = rs.fit_transform(X0)
        X1s = rs.transform(X1)
        
        pd00 = pairwise_distances(X0s, X0s) # train set
        G0 = f(pd00) # train gram matrix
        pd10 = pairwise_distances(X1s, X0s) # test set
        G1 = f(pd10) # test gram matrix
        
        svc = SVC(C=self.C, kernel='precomputed')
        svc.fit(G0, y0)
        yp = svc.predict(G1)
        
        return accuracy_score(y1, yp), x

def main():    
    for p in product(CS, DEGREES, DSETS, NS_SPLITS, TEST_SIZES, EV_CHECKS, IVT_CHECKS, SEEDS, POPS_SIZE, NS_GEN):
        C, degree, dset, n_splits, test_size, ev_check, ivt_check, seed, pop_size, n_gen = p
                    
        save_dir = os.path.join(f'exec-{n_splits}-{test_size}-{ev_check}-{ivt_check}-{pop_size}-{n_gen}', dset, str(C), str(degree))
        hist_path = os.path.join(save_dir, f'hist-{seed}.joblib')
        
        if os.path.exists(hist_path): continue
        
        with ThreadPool(processes=min(pop_size, 61)) as pool:
            print(f'Execution for {save_dir}')
            
            #runner = StarmapParallelization(pool.starmap)

            rkfp = RKFP(
                C=C, degree=degree, dset=dset,
                n_splits=n_splits, test_size=test_size,
                ev_check=ev_check, ivt_check=ivt_check, seed=seed#, elementwise_runner=runner
            )
            algo = DE(pop_size=pop_size, variant='DE/rand/1/bin', CR=0.9, dither='vector', jitter=False)
            resp = minimize(rkfp, algo, termination=('n_gen', n_gen), verbose=True, save_history=True)
            
            print(f'execution time: {resp.exec_time}, best X: {resp.X}')
            
            os.makedirs(save_dir, exist_ok=True)
            joblib.dump(resp.history, hist_path)

if __name__ == '__main__': main()