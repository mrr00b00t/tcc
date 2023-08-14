import time

from itertools import product

import numpy as np

from pmlb import fetch_data

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

import pandas as pd


def job2bdone(o):
    
    dset, test_size, seed, C, gamma = o
    
    dset_X, dset_y = fetch_data(dataset_name=dset, return_X_y=True, local_cache_dir='datasets')
    X0, X1, y0, y1 = train_test_split(dset_X, dset_y, test_size=test_size, random_state=seed, shuffle=True)
    
    rs = RobustScaler()
    X0s = rs.fit_transform(X0)
    X1s = rs.transform(X1)
    
    svc = SVC(C=C, kernel='rbf', gamma=gamma)
    svc.fit(X0s, y0)
    yp = svc.predict(X1s)
    
    acc = accuracy_score(y1, yp)
    
    print(dset, test_size, seed, C, gamma, acc)
    
    return dset, test_size, seed, C, gamma, acc

from multiprocessing import Pool

def main():
    
    DSETS = ['pima']
    TEST_SIZES = [0.2]
    SEEDS = [
       27557, 96341, 81510, 32707, 25176,  2554, 74824, 29361, 32247,
       41262, 28707, 88277, 44628,  7849, 30280, 81008, 69217, 59955,
       81265, 87455,  8707, 49946, 62014, 56900, 62529, 37215, 77929,
        9927, 49245, 36276, 85247, 21063, 56172, 67838, 71788, 38034,
       14864, 42555, 26458,  6724, 55455, 99744, 35434, 44828, 28629,
       70933, 82384, 66924, 89781, 94639, 84492, 44180, 25088, 42826,
        1310, 24549, 57876, 19679,  3974,  8625
    ]
    C_VALUES = [2**-5, 2**-3, 2**-1, 2**1, 2**3, 2**5, 2**7, 2**9, 2**11]
    GAMMAS = np.geomspace(start=1e-8, stop=1e4, num=40*50)

    inputs = list(product(DSETS, TEST_SIZES, SEEDS, C_VALUES, GAMMAS))
    
    data = {
        'dset': [], 'test_size': [], 'seed': [], 'C': [], 'gamma': [], 'acc': []
    }
    
    with Pool(processes=16) as p:
        
        tn = time.perf_counter()
        results = p.map(job2bdone, inputs)
        td = time.perf_counter()
        
        print(f'time taken: {td-tn}')
        
        for result in results:
            dset, test_size, seed, C, gamma, acc = result
            data['dset'].append(dset)
            data['test_size'].append(test_size)
            data['seed'].append(seed)
            data['C'].append(C)
            data['gamma'].append(gamma)
            data['acc'].append(acc)
            
        pd.DataFrame(data=data).to_csv('dados-rbf2.csv', sep=';', index=False)
    
if __name__ == '__main__': main()