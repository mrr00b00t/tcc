import numpy as np
import time
import pandas as pd
import os
import joblib
from tqdm import tqdm
from script import RKFP
from multiprocessing import Pool as ProcessPool
from functools import reduce


def job2bdone(filename):
    
    try:
        N_SPLITS, TEST_SIZE = 4, 0.20
        EV_CHECK, IVT_CHECK = False, True
        
        _, dataset, C, degree, hist_seed = filename.split(os.sep)
        _, seed = hist_seed.replace('.joblib', '').split('-')
        
        C = float(C)
        degree = int(degree)
        seed = int(seed)
        
        rkfp = RKFP(
            C=C, degree=degree, dset=dataset,
            n_splits=N_SPLITS, test_size=TEST_SIZE,
            ev_check=EV_CHECK, ivt_check=IVT_CHECK, seed=seed
        )
        
        hist = joblib.load(filename)
        acc_xs = [rkfp._evaluate2(x=x) for x in hist[-1].pop.get('X')]
        accuracy = np.max([u[0] for u in acc_xs])
        
        print(dataset, degree, C, seed)
        print(accuracy)
        
        return (dataset, C, degree, accuracy, seed)
    except Exception as e:
        print(f'deu ruim {filename}')


def main():
    
    folder = 'nothing2'
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
    data['accuracy'] = list()
    
    begin = time.perf_counter()

    with ProcessPool(processes=os.cpu_count()-1) as pool:
        results = pool.map(job2bdone, files)
        
        for result in results:
            dataset, C, degree, accuracy, seed = result
            data['C'].append(C)
            data['seed'].append(seed)
            data['dataset'].append(dataset)
            data['degree'].append(degree)
            data['accuracy'].append(accuracy)
    
    end = time.perf_counter()
    print(f'done in {end-begin} seconds')
    
    pd.DataFrame(data=data).to_csv(
        f'{folder}.csv', index=False, sep=';'
    )
            
if __name__ == '__main__':
    main()