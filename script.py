import os
from random import randint, sample
import joblib

from multiprocessing.pool import ThreadPool
from multiprocessing import Pool as ProcessPool

import numpy as np
from pmlb import fetch_data
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, pairwise_distances

from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.core.problem import ElementwiseProblem, StarmapParallelization

from itertools import product

# RationalKernelFunctionProblem
class RKFP(ElementwiseProblem):
    def __init__(self, C, degree, dset, n_splits, test_size, ev_check, ivt_check, seed, **kwargs):
        assert isinstance(degree, int) and C > 0.0, 'degree must be integer and must be strictly greater than zero'
        assert isinstance(degree, int) and (3 <= degree <= 7), 'degree must be integer and must be in [3, 5]'
        assert isinstance(dset, str), 'dataset name must be str'
        assert isinstance(n_splits, int) and (3 <= n_splits <= 5), 'n_splits for k fold must be int and must be in [3, 5]'
        assert isinstance(test_size, float) and (0.0 < test_size < 1.0), 'fraction of test set must be float and must be in (0.0, 1.0)'
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
        
        max_pd = -1
        _bas, _itr = list(), list()
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=randint(2, 99999))
        for train_index, valid_index in kf.split(Xns):
            
            X0s, y0 = Xns[train_index], yn[train_index]
            X1s, y1 = Xns[valid_index], yn[valid_index]
            
            pd00 = pairwise_distances(X0s, X0s)
            G0 = f(pd00) # train gram matrix
            pd10 = pairwise_distances(X1s, X0s)
            G1 = f(pd10) # valid gram matrix
            
            max_pd = max(max_pd, pd00.max(), pd10.max())
            
            svc = SVC(C=self.C, kernel='precomputed')
            svc.fit(G0, y0)
            yp = svc.predict(G1)
            
            _bas.append(balanced_accuracy_score(y1, yp))
            _itr.append(np.mean(svc.n_iter_))
        
        H0 = 0.0
        if self.ev_check:
            Xes = sample(Xns.tolist(), k=5)
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
        
        bas = np.mean(_bas)
        itr = np.mean(_itr)
        
        alpha = 0.025
        out['F'] = 1 - bas + alpha * np.log(np.mean(itr))
        
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
    SEEDS = [
       27557, 96341, 81510, 32707, 25176,  2554, 74824, 29361, 32247,
       41262, 28707, 88277, 44628,  7849, 30280, 81008, 69217, 59955,
       81265, 87455,  8707, 49946, 62014, 56900, 62529, 37215, 77929,
        9927, 49245, 36276, 85247, 21063, 56172, 67838, 71788, 38034,
       14864, 42555, 26458,  6724, 55455, 99744, 35434, 44828, 28629,
       70933, 82384, 66924, 89781, 94639, 84492, 44180, 25088, 42826,
        1310, 24549, 57876, 19679,  3974,  8625
    ]
    CS = [2**-5, 2**-3, 2**-1, 2**1, 2**3, 2**5, 2**7, 2**9, 2**11]
    DEGREES = [5, 4, 3]
    DSETS = ['pima']
    POP_SIZE, N_GEN = 40, 50
    N_SPLITS, TEST_SIZE = 4, 0.20
    
    for p in product(CS, DEGREES, DSETS, SEEDS):
        C, degree, dset, seed = p
                    
        save_dir = os.path.join('nothing2', dset, str(C), str(degree))
        plot_path = os.path.join(save_dir, f'plot-{seed}.svg')
        hist_path = os.path.join(save_dir, f'hist-{seed}.joblib')
        
        if os.path.exists(plot_path) and os.path.exists(hist_path): continue
        
        print(f'Execution for {save_dir}')
        with ThreadPool(processes=61) as pool:
            runner = StarmapParallelization(pool.starmap)

            rkfp = RKFP(
                C=C, degree=degree, dset=dset,
                n_splits=N_SPLITS, test_size=TEST_SIZE,
                ev_check=False, ivt_check=False, seed=seed, elementwise_runner=runner
            )
            algo = DE(pop_size=POP_SIZE, variant='DE/rand/1/bin', CR=0.9, dither='vector', jitter=False)
            resp = minimize(rkfp, algo, termination=('n_gen', N_GEN), verbose=True, save_history=True)
            
            print(f'execution time: {resp.exec_time}, best X: {resp.X}')
            
            acc_xs = [rkfp._evaluate2(x=x) for x in resp.pop.get('X')]
            acc_xs = sorted(acc_xs, key=lambda t: t[0], reverse=True)
            accs = [acc for acc, _ in acc_xs]
            m, s = np.median(accs), np.std(accs)
            m = round(100*m, 2)
            s = round(100*s, 2)
            
            X, _ = fetch_data(dset, return_X_y=True, local_cache_dir='datasets')
            Xs = RobustScaler().fit_transform(X)
            max_pd = pairwise_distances(Xs, Xs).max()
            pd_xrange = np.linspace(0, max_pd, 1000)
            
            min_f, max_f = np.inf, -np.inf
            
            for i, (acc, x) in enumerate(acc_xs[:5]):
                print(f'#{i} best test result', (acc, x))
                
                num, den = rkfp.num_den(x)
                f = lambda x: num(x) / den(x)
                y = f(pd_xrange)
                min_f = min(min(y), min_f)
                max_f = max(max(y), max_f)
                label = f'{round(acc*100, 2)}%'
                plt.plot(pd_xrange, y, label=label)
            
            plt.xlabel('$d(X_i, X_j)$')
            plt.ylabel('$K(X_i, X_j) = (f ∘ d) (X_i, X_j)$')
            plt.suptitle('Melhores Funções de Kernel por Acurácia no Conjunto de Teste')
            plt.title(f'{dset}, $C={C}$, $d={degree}$, $acc_m={m}$, $acc_s={s}$')
            plt.xticks(np.arange(0, max_pd + 1e-8, max_pd / 10))
            plt.yticks(np.arange(min_f, max_f + 1e-8, (max_f - min_f) / 10))
            plt.legend()
            
            ## SAVING FILES
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(plot_path)
            plt.clf()
            plt.close()
            joblib.dump(resp.history, hist_path)

if __name__ == '__main__': main()