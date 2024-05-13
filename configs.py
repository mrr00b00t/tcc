PREFIX = 'bas6'
CS = [8, 32]#[0.03125, 0.125, 0.5, 2, 8, 32, 128, 512, 2048]
MIN_GAMMA = 1e-8
MAX_GAMMA = 1e4
NS_COEFS = [4]
DSETS = ['pima']
NS_SPLITS = [4]
TEST_SIZES = [0.2]
SEEDS = [
    38016, 90168, 14688, 95474, 17620,
    5076, 29973, 6358, 19338, 49630,
    65029, 87824, 27595, 63475, 69261,
    11787, 98293, 88899, 47026, 36568,
    64176, 40883, 92001, 98218, 20468,
    82324, 18967, 59351, 25725, 85684
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