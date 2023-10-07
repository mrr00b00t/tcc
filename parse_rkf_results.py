import os
import time
import joblib
import numpy as np
import pandas as pd


from rkf import RKFP
from functools import reduce
from multiprocessing import Pool as ProcessPool


def job2bdone(filename):
    
    try:
        algo_params, dataset, C, n_coefs, hist_seed = filename.split(os.sep)
        _, n_splits, test_size, n_pop, n_gen = algo_params.split('-')
        _, seed = hist_seed.replace('.joblib', '').split('-')
        
        
        seed = int(seed)
        C = float(C)
        n_coefs = int(n_coefs)
        n_splits = int(n_splits)
        test_size = float(test_size)
        n_pop = int(n_pop)
        n_gen = int(n_gen)
        
        rkfp = RKFP(
            C=C, n_coefs=n_coefs, dset=dataset,
            n_splits=n_splits, test_size=test_size, seed=seed
        )
        
        hist = joblib.load(filename)
        valid_metrics = np.array(list(zip(hist[-1].pop.get('valid_bas'), hist[-1].pop.get('valid_itr'), hist[-1].pop.get('valid_nsv'))))
        test_metrics = np.array([rkfp._evaluate2(x=x) for x in hist[-1].pop.get('X')])
        
        best_X_argmin = np.argmin(hist[-1].pop.get('F'))
        
        F = hist[-1].pop.get('F')[best_X_argmin][0] * -1
        X_valid_bas, X_valid_itr, X_valid_nsv = valid_metrics[best_X_argmin]
        X_test_bas, X_test_itr, X_test_nsv = test_metrics[best_X_argmin]
        
        print('done:', filename)
        
        return (
            dataset, seed, C, n_coefs, F,
            X_valid_bas, X_valid_itr, X_valid_nsv,
            X_test_bas, X_test_itr, X_test_nsv,
        )
    except Exception as e:
        print(f'deu ruim {filename}', e)
        return None


def main():
    
    folder = 'basrkf-5-0.2-30-40'
    files = os.walk(folder)
    files = filter(lambda x: not x[1], files)
    files = map(lambda x: (x[0], list(filter(lambda y: 'hist' in y, x[2]))), files)
    files = map(lambda x: list(map(lambda y: os.path.join(x[0], y), x[1])), files)
    files = reduce(lambda x,y: list(list(x) + list(y)), files)
    files = list(files)
    
    print('length of files:', len(files))

    data = dict()
    data['dataset'] = list()
    data['seed'] = list()
    data['C'] = list()
    data['n_coefs'] = list()
    data['F'] = list()
    data['X_valid_bas'] = list()
    data['X_valid_itr'] = list()
    data['X_valid_nsv'] = list()
    data['X_test_bas'] = list()
    data['X_test_itr'] = list()
    data['X_test_nsv'] = list()
    
    begin = time.perf_counter()

    with ProcessPool(processes=os.cpu_count()) as pool:
        results = pool.map(job2bdone, files)
        
        for result in results:
            dataset, seed, C, n_coefs, F, X_valid_bas, X_valid_itr, X_valid_nsv, X_test_bas, X_test_itr, X_test_nsv = result
            
            data['dataset'].append(dataset)
            data['seed'].append(seed)
            data['C'].append(C)
            data['n_coefs'].append(n_coefs)
            data['F'].append(F)
            data['X_valid_bas'].append(X_valid_bas)
            data['X_valid_itr'].append(X_valid_itr)
            data['X_valid_nsv'].append(X_valid_nsv)
            data['X_test_bas'].append(X_test_bas)
            data['X_test_itr'].append(X_test_itr)
            data['X_test_nsv'].append(X_test_nsv)
    
    end = time.perf_counter()
    print(f'done in {end-begin} seconds')
    
    pd.DataFrame(data=data).to_csv(
        f'{folder}.csv', index=False, sep=';'
    )
            
if __name__ == '__main__': main()