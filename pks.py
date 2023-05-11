import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", category=ConvergenceWarning)

import joblib
import numpy as np
from random import randint
from pmlb import fetch_data
from functools import partial

from multiprocessing.pool import ThreadPool

from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import pairwise_distances, accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVC

from pymoo.optimize import minimize
from pymoo.util.display.column import Column
from pymoo.operators.sampling.lhs import LHS
from pymoo.algorithms.soo.nonconvex.de import DE, SingleObjectiveOutput
from pymoo.core.problem import ElementwiseProblem, StarmapParallelization


class CustomOutputClass(SingleObjectiveOutput):
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.va_max = Column("va_max", width=8)
        self.va_mean = Column("va_mean", width=8)
        self.vi_mean = Column("vi_mean", width=8)
        self.X_mean = Column("X_mean", width=8)
        self.columns += [self.va_max, self.va_mean, self.vi_mean, self.X_mean]
        
    def update(self, algorithm, *args, **kwargs):
        super().update(algorithm, *args, **kwargs)
        self.va_max.set(np.max(algorithm.pop.get("test_metrics")[:, 0]))
        self.va_mean.set(np.mean(algorithm.pop.get("test_metrics")[:, 0]))
        self.vi_mean.set(np.mean(algorithm.pop.get("test_metrics")[:, 2]))
        self.X_mean.set(np.mean(algorithm.pop.get("X")))

class PKSProblem(ElementwiseProblem):

    def __init__(self, **kwargs):
        self.DSET = kwargs.get('DSET')
        self.C = kwargs.get('C')
        self.APPROX = kwargs.get('APPROX')
        self.DEGREE = kwargs.get('DEGREE')
        self.VALID_SIZE = kwargs.get('VALID_SIZE')
        self.TEST_SIZE = kwargs.get('TEST_SIZE')
        self.SEED = kwargs.get('SEED')
        
        super().__init__(
            n_var=2*self.DEGREE,
            n_obj=1, n_ieq_constr=0, n_eq_constr=2,
            xl=-1.0, xu=+1.0,
            elementwise_evaluation=True, **kwargs
        )

    def num_den_f(self, x) -> callable:
        w = x.copy() / x[-1]
        coefs0, coefs1 = w[:self.DEGREE], w[self.DEGREE:]

        num, den = self.APPROX(coefs0), self.APPROX(coefs1)

        return {'num': num, 'den': den, 'f': lambda r: num(r) / den(r)}
    
    def custom_kernel(self, f):
        def func(X, Y, f): return f(pairwise_distances(X, Y) / np.sqrt(X.shape[1]))
        return partial(func, f=f)
    
    def train_test_svc(self, f, Xa, Xb, ya, yb):
    
        X0, X1, y0, y1 = Xa.copy(), Xb.copy(), ya.copy(), yb.copy()
        slc = QuantileTransformer(n_quantiles=X0.shape[0])
        X0s = slc.fit_transform(X0)
        X1s = slc.transform(X1)
        svc = SVC(C=self.C, kernel=self.custom_kernel(f=f))
        svc.fit(X0s, y0)
        y1p = svc.predict(X1s)

        return accuracy_score(y1, y1p), balanced_accuracy_score(y1, y1p), np.mean(svc.n_iter_)
    
    def split_and_train_valid_svc(self, f, X, y):
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=self.VALID_SIZE, random_state=randint(2, 999999))
        return self.train_test_svc(f, X_train, X_valid, y_train, y_valid)
        
    def ivt_constraint(self, den):
        y = den(np.random.uniform(low=0, high=1, size=1000))
        if np.any(y < 0) and np.any(y > 0):
            return 1.0
        return 0.0
    
    def ev_constraint(self, dim, f, tol=1e-8):
        A = np.random.uniform(low=0, high=1, size=(dim, dim))
        G = self.custom_kernel(f=f)(A, A)
        if np.all(np.linalg.eigvalsh(G) + tol > 0):
            return 0.0
        return 1.0    
    
    def _evaluate(self, x, out, *args, **kwargs):
        pa = self.num_den_f(x)
        _, D, f = pa['num'], pa['den'], pa['f']
        
        DX, Dy = fetch_data(self.DSET, return_X_y=True, local_cache_dir='datasets')        
        X0, X2, y0, y2 = train_test_split(DX, Dy, test_size=self.TEST_SIZE, random_state=self.SEED)
        
        train_acc, train_bacc, train_iter = self.split_and_train_valid_svc(f, X0, y0)
        test_acc, test_bacc, test_iter = self.train_test_svc(f, X0, X2, y0, y2)
        
        c_eq_0 = self.ivt_constraint(den=D)
        c_eq_1 = self.ev_constraint(dim=X0.shape[1], f=f)
        
        out["F"] = 1 - train_bacc +  0.005 * np.log(train_iter)
        out["H"] = [c_eq_0, c_eq_1]
        
        out['train_metrics'] = (train_acc, train_bacc, train_iter)
        out['test_metrics'] = (test_acc, test_bacc, test_iter)

class RationalKernelSearch:
    def __init__(self, dset, C, approx, degree, valid_size, test_size, pop_size, n_gen, seed):       
        self.DSET = dset
        self.C = C
        self.APPROX = approx
        self.DEGREE = degree
        self.VALID_SIZE = valid_size
        self.TEST_SIZE = test_size
        self.POP_SIZE = pop_size
        self.N_GEN = n_gen
        self.SEED = seed
    
    def run(self):
                
        print(f'DSET: {self.DSET}, SEED: {self.SEED}, C: {self.C}, DEGREE: {self.DEGREE}')
        
        with ThreadPool(processes=61) as pool:
            runner = StarmapParallelization(pool.starmap)
            
            pksp = PKSProblem(
                DSET=self.DSET, C=self.C, APPROX=self.APPROX, DEGREE=self.DEGREE,
                VALID_SIZE=self.VALID_SIZE, TEST_SIZE=self.TEST_SIZE,
                SEED=self.SEED, elementwise_runner=runner
            )
            
            algorithm = DE(
                pop_size=self.POP_SIZE, variant="DE/rand/1/bin",
                CR=0.9, dither="vector", jitter=False
            )
            
            res = minimize(pksp, algorithm, termination=("n_gen", self.N_GEN), verbose=True, save_history=True, output=CustomOutputClass())
            
            joblib.dump(res.history, f'results\\{self.DSET}-{self.SEED}-{self.C}-{self.DEGREE}-v2.joblib')
        
        print('Execution time:', res.exec_time)
        print('Best solution:', res.X, res.F)
        
        arg = np.argsort(res.pop.get('test_metrics')[:, 0])[-1]
        x = res.pop.get('X')[arg]
        acc = res.pop.get('test_metrics')[arg][0]
            
        return acc, x, pksp.num_den_f(x)['f'], pksp.C, pksp.SEED
            