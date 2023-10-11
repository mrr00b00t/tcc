PREFIX = 'bas'
CS = [0.03125, 0.125, 0.5, 2, 8, 32, 128, 512, 2048]
MIN_GAMMA = 1e-8
MAX_GAMMA = 1e4
NS_COEFS = [3, 4, 5]
DSETS = ['pima']
NS_SPLITS = [5]
TEST_SIZES = [0.2]
SEEDS = [
    37837, 15431, 51869, 19822, 32325,  3325,
    46810, 89597, 82876,  1718, 63495, 44271,
    71476, 17892, 37448, 71477, 50575, 59520,
    54551, 18176, 68520, 95674, 87498, 80573,
    35634, 95531, 98297, 29155, 10070, 37134,
    19593, 52306, 38297, 68043, 70134, 39750,
     3786, 57556, 13652, 57967, 71675, 96728,
    48499, 23128, 41856, 79959, 34488, 91937,
    87605, 47079, 85035, 78578, 13150, 75916,
    28922, 49870, 74923, 74740, 80300, 42750
]
POPS_SIZE = [30]
NS_GEN = [40]
SVC_MAX_ITER = 1800


if __name__ == '__main__':

    for dset in DSETS:
        from pmlb import fetch_data
        
        X, y = fetch_data(dataset_name=dset, return_X_y=True, local_cache_dir='datasets')
        
        print('X shape:', X.shape)
        print('y shape', y.shape)