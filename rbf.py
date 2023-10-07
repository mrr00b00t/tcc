import os, time
import numpy as np
import pandas as pd

from pmlb import fetch_data
from itertools import product
from multiprocessing import Pool as ProcessPool

from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold

from configs import PREFIX, CS, MIN_GAMMA, MAX_GAMMA, DSETS, NS_SPLITS, TEST_SIZES, SEEDS, POPS_SIZE, NS_GEN, SVC_MAX_ITER

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


@ignore_warnings(category=ConvergenceWarning)
def job2bdone(_input):

    dataset, seed, C, gamma, n_splits, test_size = _input
    
    dset_X, dset_y = fetch_data(
        dataset_name=dataset,
        return_X_y=True,
        local_cache_dir='datasets'
    )

    Xv, Xt, yv, yt = train_test_split(
        dset_X, dset_y, test_size=test_size,
        shuffle=True, random_state=seed, stratify=dset_y
    )
    
    l_valid_bas, l_valid_itr, l_valid_nsv = list(), list(), list()
    
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_index, test_index in kf.split(Xv, yv):
        X0, y0 = Xv[train_index], yv[train_index]
        X1, y1 = Xv[test_index], yv[test_index]
        
        rs = StandardScaler()
        X0s = rs.fit_transform(X0)
        X1s = rs.transform(X1)
    
        svc = SVC(C=C, kernel='rbf', gamma=gamma, max_iter=SVC_MAX_ITER)
        svc.fit(X0s, y0)
        yp = svc.predict(X1s)
        
        l_valid_bas.append( balanced_accuracy_score(y1, yp) )
        l_valid_itr.append( np.mean(svc.n_iter_) )
        l_valid_nsv.append( svc.support_.shape[0] / X0s.shape[0] )
    
    valid_bas, valid_itr, valid_nsv = np.mean(l_valid_bas), np.mean(l_valid_itr), np.mean(l_valid_nsv)
    
    rs = StandardScaler()
    Xvs = rs.fit_transform(Xv)
    Xts = rs.transform(Xt)
    
    svc = SVC(C=C, kernel='rbf', gamma=gamma, max_iter=SVC_MAX_ITER)
    svc.fit(Xvs, yv)
    yp = svc.predict(Xts)
    
    test_bas, test_itr, test_nsv = balanced_accuracy_score(yt, yp), np.mean( svc.n_iter_ ), svc.support_.shape[0] / Xvs.shape[0]
    
    print(dataset, seed, C, gamma)
    
    return dataset, seed, C, gamma, valid_bas, valid_itr, valid_nsv, test_bas, test_itr, test_nsv

def main():
    
    for n_splits, test_size, pop_size, n_gen in product(NS_SPLITS, TEST_SIZES, POPS_SIZE, NS_GEN):
        
        GAMMAS = np.geomspace(start=MIN_GAMMA, stop=MAX_GAMMA, num=pop_size * n_gen)
        
        data = dict()
        data['dataset'] = list()
        data['seed'] = list()
        data['C'] = list()
        data['gamma'] = list()
        data['valid_bas'] = list()
        data['valid_itr'] = list()
        data['valid_nsv'] = list()
        data['test_bas'] = list()
        data['test_itr'] = list()
        data['test_nsv'] = list()
        
        inputs = product(DSETS, SEEDS, CS, GAMMAS, [n_splits], [test_size])
        
        with ProcessPool(processes=os.cpu_count()) as p:
            
            tn = time.perf_counter()
            results = p.map(job2bdone, inputs)
            td = time.perf_counter()
            
            print(f'time taken: {td-tn}')
            
            for result in results:
                dataset, seed, C, gamma, valid_bas, valid_itr, valid_nsv, test_bas, test_itr, test_nsv = result
                
                data['dataset'].append(dataset)
                data['seed'].append(seed)
                data['C'].append(C)
                data['gamma'].append(gamma)
                data['valid_bas'].append(valid_bas)
                data['valid_itr'].append(valid_itr)
                data['valid_nsv'].append(valid_nsv)
                data['test_bas'].append(test_bas)
                data['test_itr'].append(test_itr)
                data['test_nsv'].append(test_nsv)
                
            pd.DataFrame(
                data=data
            ).to_csv(
                f'{PREFIX}rbf-{n_splits}-{test_size}-{pop_size}-{n_gen}.csv',
                sep=';',
                index=False
            )

if __name__ == '__main__': main()