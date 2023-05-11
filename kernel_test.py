from multiprocessing import Pool

from pmlb import fetch_data

from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd

from itertools import product

import time


def svm_test(dset, SEED, C, TEST_SIZE=0.25):
    X,y = fetch_data(dataset_name=dset, return_X_y=True, local_cache_dir='datasets')
    
    X0, X2, y0, y2 = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
    
    transformer = QuantileTransformer(n_quantiles=X0.shape[0])
    X0s = transformer.fit_transform(X0)
    X2s = transformer.transform(X2)
    
    svc = SVC(C=C, kernel='linear')
    svc.fit(X0s, y0)
    y2p = svc.predict(X2s)
    acc = accuracy_score(y2, y2p)
    
    return (dset, SEED, C, acc, np.mean(svc.n_iter_))

def main():
    SEEDs = [
        941851, 936159, 118334, 108183, 479829, 773980, 910215, 895438,
        228808, 119632, 745699, 680352, 717830, 988865, 842375, 750994,
        719354, 329909, 599749, 328876, 161269, 422522, 858762, 195498,
        167992, 452798, 928982, 930767, 281191, 837153]
    
    Cs = [0.03125, 0.125, 0.5, 2.0, 8.0, 32.0, 128.0, 512.0, 2048.0]

    dsets = ['pima', 'ionosphere']
    
    gammas = np.linspace(1.e-9, 1.e3, 30*50)
    
    
    begin = time.perf_counter()
    with Pool(processes=16) as pool:
        r = pool.starmap(svm_test, product(dsets, SEEDs, Cs))
    end = time.perf_counter()
    
    r = list(r)
    arr = np.array(r)
    
    data = dict()
    data['dset'] = arr[:, 0]
    data['SEED'] = arr[:, 1]
    data['C'] = arr[:, 2]
    data['acc'] = arr[:, 3]
    data['iter'] = arr[:, 4]
    
    pd.DataFrame(data=data).to_csv('kernel_test_linear.csv', index=False)
    
    print('Total time: {}'.format(end-begin))

if __name__ == '__main__':
    main()