from pks import RationalKernelSearch

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def poly3(c):
    return lambda x: (c[0]*x + c[1])*x + c[2]

def poly4(c):
    return lambda x: ((c[0]*x + c[1])*x + c[2])*x + c[3]

def poly5(c):
    return lambda x: (((c[0]*x + c[1])*x + c[2])*x + c[3])*x + c[4]

def poly6(c):
    return lambda x: ((((c[0]*x + c[1])*x + c[2])*x + c[3])*x + c[4])*x + c[5]

def poly7(c):
    return lambda x: (((((c[0]*x + c[1])*x + c[2])*x + c[3])*x + c[4])*x + c[5])*x + c[6]

def main():
    
    SEEDs = [
        941851, 936159, 118334, 108183, 479829, 773980, 910215, 895438,
        228808, 119632, 745699, 680352, 717830, 988865, 842375, 750994,
        719354, 329909, 599749, 328876, 161269, 422522, 858762, 195498,
        167992, 452798, 928982, 930767, 281191, 837153]
    
    Cs = [0.03125, 0.125, 0.5, 2.0, 8.0, 32.0, 128.0, 512.0, 2048.0]
    
    dsets = ['glass2', 'breast_cancer_wisconsin']
    tests = [('poly3', poly3, 3), ('poly4', poly4, 4),('poly5', poly5, 5), ('poly6', poly6, 6), ('poly7', poly7, 7)]
    
    for dset in dsets:
        for pstr, pfunc, pdegree in tests:
            data = {
                'seed': list(),
                'C': list(),
                'x': list(),
                'acc': list(),
            }
            
            for _C in Cs:
                for _seed in SEEDs:
                    rksp = RationalKernelSearch(dset=dset, C=_C, approx=pfunc, degree=pdegree, valid_size=0.33, test_size=0.25, pop_size=30, n_gen=50, seed=_seed)
                    acc, x, f, c, s = rksp.run()
                    
                    data['acc'].append(acc)
                    data['x'].append(x)
                    data['C'].append(c)
                    data['seed'].append(s)
                        
            pd.DataFrame(data=data).to_csv(f'{dset}-{pstr}-30-50-v2.csv', index=False)
            
if __name__ == '__main__':
    main()