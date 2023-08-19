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
        algo_params, dataset, C, degree, hist_seed = filename.split(os.sep)
        _, n_splits, test_size, ev_check, ivt_check, n_pop, n_gen = algo_params.split('-')
        _, seed = hist_seed.replace('.joblib', '').split('-')
        
        C = float(C)
        degree = int(degree)
        seed = int(seed)
        n_splits = int(n_splits)
        test_size = float(test_size)
        ev_check = bool(ev_check)
        ivt_check = bool(ivt_check)
        n_pop = int(n_pop)
        n_gen = int(n_gen)
        
        rkfp = RKFP(
            C=C, degree=degree, dset=dataset,
            n_splits=n_splits, test_size=test_size,
            ev_check=ev_check, ivt_check=ivt_check, seed=seed
        )
        
        hist = joblib.load(filename)
        acc_xs = [rkfp._evaluate2(x=x) for x in hist[-1].pop.get('X')]
        best_valid_acc = np.max([u[0] for u in acc_xs])
        Md_valid_acc = np.median([u[0] for u in acc_xs])
        Mn_valid_acc = np.mean([u[0] for u in acc_xs])
        
        
        print(dataset, degree, C, round(100*best_valid_acc, 2), round(100*Md_valid_acc, 2), round(100*Mn_valid_acc, 2), seed)
        
        return (dataset, C, degree, best_valid_acc, Md_valid_acc, Mn_valid_acc, seed)
    except Exception as e:
        print(f'deu ruim {filename}', e)


def main():
    
    folder = 'exec-5-0.2-False-False-30-50'
    files = os.walk(folder)
    files = filter(lambda x: not x[1], files)
    files = filter(lambda x: 'pima' in x[0], files)
    files = map(lambda x: (x[0], list(filter(lambda y: 'hist' in y, x[2]))), files)
    files = map(lambda x: list(map(lambda y: os.path.join(x[0], y), x[1])), files)
    files = reduce(lambda x,y: list(list(x) + list(y)), files)
    files = list(files)
    
    print('length of files:', len(files))

    data = dict()
    data['C'] = list()
    data['seed'] = list()
    data['dataset'] = list()
    data['degree'] = list()
    data['best_valid_acc'] = list()
    data['Md_valid_acc'] = list()
    data['Mn_valid_acc'] = list()
    
    begin = time.perf_counter()

    with ProcessPool(processes=os.cpu_count()) as pool:
        results = pool.map(job2bdone, files)
        
        for result in results:
            dataset, C, degree, best_valid_acc, Md_valid_acc, Mn_valid_acc, seed = result
            data['C'].append(C)
            data['seed'].append(seed)
            data['dataset'].append(dataset)
            data['degree'].append(degree)
            data['best_valid_acc'].append(best_valid_acc)
            data['Md_valid_acc'].append(Md_valid_acc)
            data['Mn_valid_acc'].append(Mn_valid_acc)
    
    end = time.perf_counter()
    print(f'done in {end-begin} seconds')
    
    pd.DataFrame(data=data).to_csv(
        f'{folder}.csv', index=False, sep=';'
    )
            
if __name__ == '__main__': main()