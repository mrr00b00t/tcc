import os
import time
import numpy as np
import pandas as pd

from pmlb import fetch_data
from itertools import product
from multiprocessing import Pool as ProcessPool

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

from configs import CS, DSETS, TEST_SIZES, SEEDS, POPS_SIZE, NS_GEN, MIN_GAMMA, MAX_GAMMA


def job2bdone(input):
    
    C, dset, test_size, seed, gamma = input
    
    dset_X, dset_y = fetch_data(dataset_name=dset, return_X_y=True, local_cache_dir='datasets')
    X0, X1, y0, y1 = train_test_split(dset_X, dset_y, test_size=test_size, random_state=seed, shuffle=True)
    
    rs = RobustScaler()
    X0s = rs.fit_transform(X0)
    X1s = rs.transform(X1)
    
    svc = SVC(C=C, kernel='rbf', gamma=gamma)
    svc.fit(X0s, y0)
    yp = svc.predict(X1s)
    
    test_acc = accuracy_score(y1, yp)
    
    print(C, dset, test_size, seed, gamma, test_acc)
    
    return C, dset, test_size, seed, gamma, test_acc

def main():
    
    for pop_size, n_gen in product(POPS_SIZE, NS_GEN):
        data = dict()
        
        data['C'] = list()
        data['dset'] = list()
        data['test_size'] = list()
        data['seed'] = list()
        data['gamma'] = list()
        data['test_acc'] = list()
        
        GAMMAS = np.geomspace(start=MIN_GAMMA, stop=MAX_GAMMA, num=pop_size * n_gen)
        
        inputs = product(CS, DSETS, TEST_SIZES, SEEDS, GAMMAS)
        
        with ProcessPool(processes=os.cpu_count()) as p:
            
            tn = time.perf_counter()
            results = p.map(job2bdone, inputs)
            td = time.perf_counter()
            
            print(f'time taken: {td-tn}')
            
            for result in results:
                C, dset, test_size, seed, gamma, test_acc = result
                
                data['C'].append(C)
                data['dset'].append(dset)
                data['test_size'].append(test_size)
                data['seed'].append(seed)
                data['gamma'].append(gamma)
                data['test_acc'].append(test_acc)
                
            pd.DataFrame(
                data=data
            ).to_csv(
                f'exec-rbf-{test_size}-{pop_size}-{n_gen}.csv',
                sep=';',
                index=False
            )
    
if __name__ == '__main__': main()