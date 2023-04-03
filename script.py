import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import  balanced_accuracy_score

from matplotlib import pyplot as plt

from pymoo.core.problem import ElementwiseProblem
from pmlb import fetch_data

import script_configs as configs


class MyProblem(ElementwiseProblem):

    def __init__(self, **kwargs):
        
        X, y = fetch_data('ionosphere', return_X_y=True, local_cache_dir='datasets')
        Xt, Xv, yt, yv = train_test_split(X, y, test_size=configs.VALIDATION_SIZE, stratify=y, random_state=configs.SEED)
        
        self.X = Xt
        self.y = yt
        self.Xv = Xv
        self.yv = yv
        
        self.C = 0.1
        self.dt_iter = 1. / self.y.shape[0]
        
        super().__init__(
            n_var=12, n_obj=2,
            n_ieq_constr=0, n_eq_constr=0,
            xl=-1.0, xu=+1.0,
            elementwise_evaluation=True,
            **kwargs
        )

    def _is_positive_semidefinite(self, A: np.ndarray, tol=1e-8) -> bool:
        
        E = np.linalg.eigvalsh(A)
        
        return np.all(E > -tol)

    def _train_test_svc(self, Xtrain, Xtest, ytrain, ytest, f) -> tuple:
        
        _Xtrain = Xtrain.copy()
        _Xtest = Xtest.copy()
        _ytrain = ytrain.copy()
        _ytest = ytest.copy()
        
        dim = _Xtrain.shape[1]
        
        # escala a entrada
        scaler  = StandardScaler()
        _Xtrain = scaler.fit_transform(_Xtrain)
        _Xtest  = scaler.transform(_Xtest)
        
        # criar classificador
        svc = SVC(C=self.C, kernel='precomputed')
        
        # calcula kernel de treino e treina classificador
        kernel_train = f(np.dot(_Xtrain, _Xtrain.T))
        kernel_train = (kernel_train + dim) / (2*dim)
        svc.fit(kernel_train, _ytrain)
        
        # calcula kernel de teste e faz inferência
        kernel_test = f(np.dot(_Xtest, _Xtrain.T))
        kernel_test = (kernel_test + dim) / (2*dim)
        ypred = svc.predict(kernel_test)
        
        # calcula métricas
        _acc = balanced_accuracy_score(_ytest, ypred)
        _iter = svc.n_iter_[0]
        
        return (_acc, _iter)

    def training_k_fold_cross_validation(self, f):
        
        skf = StratifiedKFold(n_splits=5, random_state=configs.SEED, shuffle=True)

        accs, iters = list(), list()

        for i, (train_index, test_index) in enumerate(skf.split(self.X, self.y)):
            
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            
            acc, iter = self._train_test_svc(X_train, X_test, y_train, y_test, f)
            
            accs.append(acc)
            iters.append(iter)
        
        acc_mean = np.mean(accs)
        iter_mean = np.mean(iters)
        
        return (acc_mean, iter_mean)
        
    def test_validation(self, f):
        return self._train_test_svc(self.X, self.Xv, self.y, self.yv, f)

    def _evaluate(self, x, out, *args, **kwargs):

        z = x.copy()

        N = z[:6]
        D = z[6:]

        def num(r): return ((((N[0]*r + N[1])*r + N[2])*r + N[3])*r + N[4])*r + N[5]
        def den(r): return ((((D[0]*r + D[1])*r + D[2])*r + D[3])*r + D[4])*r + D[5]
        def f(r): return num(r) / den(r)
        
        train_acc, train_iter = self.training_k_fold_cross_validation(f)
        valid_acc, valid_iter = self.test_validation(f)
        
        out["F"] = [-1 * train_acc, train_iter]
        out["valid_metrics"] = (valid_acc, valid_iter)
         
from multiprocessing.pool import ThreadPool
from pymoo.core.problem import StarmapParallelization
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.operators.sampling.lhs import LHS

def main():
    n_threads = 40
    with ThreadPool(n_threads) as pool:
        runner = StarmapParallelization(pool.starmap)

        # define the problem by passing the starmap interface of the thread pool
        problem = MyProblem(elementwise_runner=runner)
        
        ## DEFINIR RESTRIÇÃO DE NÃO TER POLOS NO INTERVALO DA FUNÇÃO
        nsga2 = NSGA2(pop_size=400,
                    eliminate_duplicates=True)
        
        de = DE(
            pop_size=250,
            sampling=LHS(),
            variant="DE/rand/1/bin",
            CR=0.3,
            dither="vector",
            jitter=False
        )

        print(50*'=')
        print('Seed:', configs.SEED)
        print('Pymoo seed', configs.PYMOO_SEED)
        res = minimize(problem, nsga2, termination=("n_gen", 25), seed=configs.PYMOO_SEED, verbose=True, save_history=False)
        print('Execution time:', res.exec_time/60.)
        print('Best solution:', res.X)
        print(50*'=')

        sx, sy, sc = list(), list(), list()

        for X in res.X:
            mp = MyProblem()
            d = {}
            mp._evaluate(X, d)
            
            
            _acc, _iter = d['F']
            _valid_acc, _valid_iter = d['valid_metrics']
            
            print((_acc, _iter), (_valid_acc, _valid_iter))
        
        plot = Scatter()
        plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
        plot.add(res.F, facecolor="none", edgecolor="red")
        plot.show()
                
    return
        
if __name__ == '__main__':
    main()