import numpy as np
from functools import partial

from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import pairwise_distances, accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVC


PADDE_N_M_DEGREE: int = 3
SVC_MAX_ITER: int = -1
SVC_C: float = 1/32
SEED: int = 42
DATASET_NAME: str = 'pima'
VALIDATION_K_SPLITS: int = 5
CURRENT_K_SPLIT: int = 0
TEST_SIZE: float = 0.25
LOW: float = -0.25
HIGH: float = +1.25

def rnd() -> int: np.random.randint(low=2, high=99999, size=1)

def is_positive_semidefinite(A, tol=1e-8) -> bool: return np.all(np.linalg.eigvalsh(A) > -tol)

def get_padde_approx_from_ndarray(z) -> callable:
    
    w = z.copy() / z[-1]
    N = w[:PADDE_N_M_DEGREE]
    D = w[PADDE_N_M_DEGREE:]
    
    num = lambda r: (N[0]*r + N[1])*r + N[2]
    den = lambda r: (D[0]*r + D[1])*r + D[2]
    
    return {'num': num, 'den': den, 'f': lambda r: num(r) / den(r)}

def custom_kernel_with_distances(X, Y, f): return f(pairwise_distances(X, Y) / np.sqrt(X.shape[1]))

def train_test_svc(f, X_train, X_test, y_train, y_test):
    
    X0, X1, y0, y1 = X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()
    slc = QuantileTransformer(n_quantiles=X0.shape[0])
    X0s = slc.fit_transform(X0)
    X1s = slc.transform(X1)
    svc = SVC(C=SVC_C, kernel=partial(custom_kernel_with_distances, f=f), max_iter=SVC_MAX_ITER)
    svc.fit(X0s, y0)
    y1p = svc.predict(X1s)

    return accuracy_score(y1, y1p), balanced_accuracy_score(y1, y1p), np.mean(svc.n_iter_)

def split_and_train_test_svc(f, X, y): return train_test_svc(f, *train_test_split(X, y, test_size=TEST_SIZE, random_state=rnd()))

def ivt_constraint(den):
    
    x = np.random.uniform(low=LOW, high=HIGH, size=10000)
    y = den(x)
    
    if np.any(y < 0) and np.any(y > 0):
        return 1.0
    return 0.0

def ev_constraint(dim, f):
    
    A = np.random.uniform(low=LOW, high=HIGH, size=(dim, dim))
    gram_matrix = custom_kernel_with_distances(A, A, f)
    
    if is_positive_semidefinite(gram_matrix):
        return 0.0
    return 1.0

def get_02_idx(arr):
    
    kfold = KFold(n_splits=VALIDATION_K_SPLITS, shuffle=True, random_state=SEED)
    splits = list(kfold.split(arr))
    return splits[CURRENT_K_SPLIT]