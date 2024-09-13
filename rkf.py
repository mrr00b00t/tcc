"""Arquivo com o código da nova abordagem para o TCC"""

import os
import time

from itertools import product
from multiprocessing.pool import Pool

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, pairwise_distances

import joblib
import numpy as np

from pmlb import fetch_data

from configs import (
    PREFIX, CS, NS_COEFS, DSETS, NS_SPLITS,
    TEST_SIZES, POPS_SIZE, NS_GEN, SEEDS,
    MAX_ITER
)


import unittest.mock


with unittest.mock.patch("pymoo.gradient.TOOLBOX", new='jax.numpy'):
    from pymoo.optimize import minimize
    from pymoo.algorithms.soo.nonconvex.de import DE
    from pymoo.core.problem import ElementwiseProblem


class RKFP(ElementwiseProblem):
    """Classe com a nova abordagem de execução para o Kernel Racional"""

    def __init__(self, **kwargs):
        self.c_param = kwargs['c_param']
        self.n_coefs = kwargs['n_coefs']
        self.dset = kwargs['dset']
        self.n_splits = kwargs['n_splits']
        self.test_size = kwargs['test_size']
        self.seed = kwargs['seed']

        super().__init__(
            n_var=2*self.n_coefs, n_obj=1, n_ieq_constr=0, n_eq_constr=0,
            xl=-1.0, xu=+1.0, elementwise_evaluation=True, **kwargs
        )

    def arr2poly(self, a):
        """Transforma um array em um polinômio"""

        if self.n_coefs == 3:
            return lambda x: (a[0]*x + a[1])*x + a[2]
        if self.n_coefs == 4:
            return lambda x: ((a[0]*x + a[1])*x + a[2])*x + a[3]
        if self.n_coefs == 5:
            return lambda x: (((a[0]*x + a[1])*x + a[2])*x + a[3])*x + a[4]

    def get_f(self, x):
        """Transforma um array em uma função racional"""

        num = self.arr2poly(x[:self.n_coefs])
        den = self.arr2poly(x[self.n_coefs:])
        return lambda y: num(y) / den(y)

    @ignore_warnings(category=ConvergenceWarning)
    def _evaluate(self, x, out, *args, **kwargs):
        f = self.get_f(x=x.copy())

        dset_x, dset_y = fetch_data(
            dataset_name=self.dset, return_X_y=True,
            local_cache_dir='datasets'
        )
        xn,  _, yn,  _ = train_test_split(
            dset_x, dset_y, test_size=self.test_size,
            random_state=self.seed, shuffle=True, stratify=dset_y
        )

        l_bas, l_itr, l_nsv = [], [], []

        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True,
            random_state=self.seed
        )
        splits = skf.split(X=xn, y=yn)
        for train_index, test_index in splits:
            x0, x1 = xn[train_index], xn[test_index]
            y0, y1 = yn[train_index], yn[test_index]

            rs = StandardScaler()
            x0s = rs.fit_transform(x0)
            x1s = rs.transform(x1)

            pd00 = pairwise_distances(x0s, x0s)
            g0 = f(pd00)  # train gram matrix
            pd10 = pairwise_distances(x1s, x0s)
            g1 = f(pd10)  # valid gram matrix

            svc = SVC(C=self.c_param, kernel='precomputed', max_iter=MAX_ITER)
            svc.fit(g0, y0)
            yp = svc.predict(g1)

            l_bas.append(balanced_accuracy_score(y_true=y1, y_pred=yp))
            l_itr.append(np.mean(svc.n_iter_))
            l_nsv.append(svc.support_.shape[0] / x0s.shape[0])

        bas = np.mean(l_bas)
        itr = np.mean(l_itr)
        nsv = np.mean(l_nsv)

        alpha = self.n_splits / yn.shape[0]
        beta = itr / MAX_ITER
        out['F'] = -1 * bas + alpha * (beta + nsv)

        out['valid_bas'], out['valid_itr'], out['valid_nsv'] = bas, itr, nsv

        t_bas, t_itr, t_nsv = self._evaluate2(x=x.copy())
        out['test_bas'], out['test_itr'], out['test_nsv'] = t_bas, t_itr, t_nsv

    @ignore_warnings(category=ConvergenceWarning)
    def _evaluate2(self, x):
        f = self.get_f(x=x.copy())

        dset_x, dset_y = fetch_data(
            dataset_name=self.dset, return_X_y=True,
            local_cache_dir='datasets'
        )
        x0, x1, y0, y1 = train_test_split(
            dset_x, dset_y, test_size=self.test_size,
            random_state=self.seed, shuffle=True, stratify=dset_y
        )

        rs = StandardScaler()
        x0s = rs.fit_transform(x0)
        x1s = rs.transform(x1)

        pd00 = pairwise_distances(x0s, x0s)
        g0 = f(pd00)  # train gram matrix
        pd10 = pairwise_distances(x1s, x0s)
        g1 = f(pd10)  # test gram matrix

        svc = SVC(C=self.c_param, kernel='precomputed', max_iter=MAX_ITER)
        svc.fit(g0, y0)
        yp = svc.predict(g1)

        bas = balanced_accuracy_score(y_true=y1, y_pred=yp)
        itr = np.mean(svc.n_iter_)
        nsv = svc.support_.shape[0] / x0s.shape[0]

        return bas, itr, nsv


def job2bdone(p):
    """Função de execução do experimento para ser paralelizada"""

    c_param, n_coefs, dset, n_splits, test_size, seed, pop_size, n_gen = p

    root = f'{PREFIX}rkf-{n_splits}-{test_size}-{pop_size}-{n_gen}'
    save_dir = os.path.join(root, dset, str(c_param), str(n_coefs))
    hist_path = os.path.join(save_dir, f'hist-{seed}.joblib')

    if os.path.exists(hist_path):
        return 'already done'

    rkfp = RKFP(
        c_param=c_param, n_coefs=n_coefs, dset=dset,
        n_splits=n_splits, test_size=test_size, seed=seed
    )
    algo = DE(
        pop_size=pop_size, variant='DE/rand/1/bin',
        CR=0.9, F=0.8, dither='vector', jitter=False
    )
    resp = minimize(
        rkfp, algo, termination=('n_gen', n_gen),
        verbose=False, save_history=True
    )

    print(
        f'done path {save_dir}',
        f'execution time: {resp.exec_time}',
        f'best X: {resp.X}',
        f'best F: {resp.F}'
    )

    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(resp.history, hist_path)

    return 'done'


def main():
    """Função principal"""

    products = product(
        CS, NS_COEFS, DSETS, NS_SPLITS, TEST_SIZES,
        SEEDS, POPS_SIZE, NS_GEN
    )

    products = list(products)

    results = []
    cpu_count = os.cpu_count() // 2
    with Pool(processes=cpu_count) as pool:
        begin_time = time.time()
        results = pool.map(job2bdone, products)
        end_time = time.time()

        print(f'total time to execute: {end_time-begin_time}')

    print(results)


if __name__ == '__main__':
    main()
