"""Arquivo de configurações para execução do código"""

PREFIX = 'multiproc2'
CS = [0.5]
MIN_GAMMA = 1e-8
MAX_GAMMA = 1e4
NS_COEFS = [3, 4, 5]
DSETS = ['pima']
NS_SPLITS = [4]
TEST_SIZES = [0.2]
SEEDS = [
    64545, 10366, 91895, 56105, 50879, 80641, 52241, 71635,
    24500, 92681, 42181, 64344, 33381, 93928, 7542, 85952
]
POPS_SIZE = [40]
NS_GEN = [30]
MAX_ITER = 1200


if __name__ == '__main__':

    for dset in DSETS:
        from pmlb import fetch_data

        X, y = fetch_data(
            dataset_name=dset,
            return_X_y=True,
            local_cache_dir='datasets'
        )

        print('X shape:', X.shape)
        print('y shape', y.shape)
