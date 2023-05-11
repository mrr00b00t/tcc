import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", category=ConvergenceWarning)

import os
import utils
import numpy as np

from multiprocessing.pool import ThreadPool

from pmlb import fetch_data
from matplotlib import pyplot as plt

from pymoo.optimize import minimize
from pymoo.util.display.column import Column
from pymoo.operators.sampling.lhs import LHS
from pymoo.algorithms.soo.nonconvex.de import DE, SingleObjectiveOutput
from pymoo.core.problem import ElementwiseProblem, StarmapParallelization


def custom_output(OutputClass):
    
    class CustomOutput(OutputClass):
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.va_max = Column("va_max", width=8)
            self.va_mean = Column("va_mean", width=8)
            self.vi_mean = Column("vi_mean", width=8)
            self.X_mean = Column("X_mean", width=8)
            self.columns += [self.va_max, self.va_mean, self.vi_mean, self.X_mean]
            
        def update(self, algorithm, *args, **kwargs):
            super().update(algorithm, *args, **kwargs)
            self.va_max.set(np.max(algorithm.pop.get("valid_metrics")[:, 0]))
            self.va_mean.set(np.mean(algorithm.pop.get("valid_metrics")[:, 0]))
            self.vi_mean.set(np.mean(algorithm.pop.get("valid_metrics")[:, 2]))
            self.X_mean.set(np.mean(algorithm.pop.get("X")))
    
    return CustomOutput()

class MyProblem(ElementwiseProblem):

    def __init__(self, **kwargs):
        super().__init__(n_var=2*utils.PADDE_N_M_DEGREE, n_obj=1, n_ieq_constr=0, n_eq_constr=2, xl=-1.0, xu=+1.0, elementwise_evaluation=True, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        pa = utils.get_padde_approx_from_ndarray(z=x.copy())
        N, D, f = pa['num'], pa['den'], pa['f']
        
        DX, Dy = fetch_data(utils.DATASET_NAME, return_X_y=True, local_cache_dir='datasets')
        idx0, idx2 = utils.get_02_idx(arr=DX)
        X0, X2, y0, y2 = DX[idx0], DX[idx2], Dy[idx0], Dy[idx2],        
        
        train_acc, train_bacc, train_iter = utils.split_and_train_test_svc(f, X0, y0)
        valid_acc, valid_bacc, valid_iter = utils.train_test_svc(f, X0, X2, y0, y2)
        
        c_eq_0 = utils.ivt_constraint(den=D)
        c_eq_1 = utils.ev_constraint(dim=X0.shape[1], f=f)
        
        out["F"] = 1 - train_bacc +  0.01 * np.log(train_iter)
        out["H"] = [c_eq_0, c_eq_1]
        
        out['train_metrics'] = (train_acc, train_bacc, train_iter)
        out['valid_metrics'] = (valid_acc, valid_bacc, valid_iter)

def main():
    n_threads = 50
    with ThreadPool(n_threads) as pool:
        runner = StarmapParallelization(pool.starmap)

        # define the problem by passing the starmap interface of the thread pool
        problem = MyProblem(elementwise_runner=runner)
        
        algorithm = DE(
            pop_size=25,
            sampling=LHS(),
            variant="DE/rand/1/exp",
            CR=0.9,
            dither="vector",
            jitter=False
        )

        print('Seed:', utils.SEED)
        res = minimize(problem, algorithm, termination=("n_gen", 40), verbose=True, save_history=True, output=custom_output(SingleObjectiveOutput))
        print('Execution time:', res.exec_time/60.)
        print('Best solution:', res.X, res.F)
        
        first_n = 5
        F_pop, X_pop = res.pop.get('F'), res.pop.get('X')
        args_sorted = np.argsort(-1 * res.pop.get('valid_metrics')[:, 0])
        args = args_sorted[:first_n]
        
        SAVE_FOLDER = 'results'
        os.makedirs(SAVE_FOLDER, exist_ok=True)
        
        
        my_Cs = list()
        
        for z in X_pop[args]:
            x = np.linspace(utils.LOW, utils.HIGH, 1000)
            y = utils.get_padde_approx_from_ndarray(z)['f'](x)
            y = (y - min(y)) / (max(y) - min(y))
            
            d = {}
            MyProblem()._evaluate(z, d)
    
            F = d['F']
            acc, bacc, iter = d['train_metrics']
            acc, bacc, iter = round(acc, 5), round(bacc, 5), round(iter, 5)
            valid_acc, valid_bacc, valid_iter = d['valid_metrics']
            valid_acc, valid_bacc, valid_iter = round(valid_acc, 5), round(valid_bacc, 5), round(valid_iter, 5)
            print(z)
            print('F:', F, 'C:', utils.SVC_C, 'acc:', acc, 'bacc:', bacc, 'iter:', iter, 'valid_acc', valid_acc, 'valid_bacc', valid_bacc, 'valid_iter', valid_iter)
            
            plt.plot(x, y, label='C={}, VA={}'.format(utils.SVC_C, round(valid_acc * 100, 2)))
            
        plt.xlim(utils.LOW, utils.HIGH)
        plt.legend()
        plt.savefig(os.path.join(SAVE_FOLDER, '-'.join([utils.DATASET_NAME, str(utils.SEED), str(utils.CURRENT_K_SPLIT), str(utils.SVC_C), '.png'])), dpi=300)
        plt.close()
        
if __name__ == '__main__':
    
    N_EXPERIMENTS = 30
    SEEDs = np.random.randint(low=2, high=99999, size=N_EXPERIMENTS)
    Cs = [0.03125, 0.125, 0.5, 2.0, 8.0, 32.0, 128.0, 512.0, 2048.0]
    
    for c in Cs:
        for s in SEEDs:
            utils.SVC_C = c
            utils.SEED = s
        
            for i in range(utils.VALIDATION_K_SPLITS):
                utils.CURRENT_K_SPLIT = i
                
                main()