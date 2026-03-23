import numpy as np
from scipy.io import loadmat
from OALC.utils import mask_types, mask_types_old

def seed_everything(seed: int):
    np.random.seed(seed)

seed_everything(42)

def capricious(X, ratio):
    MASK_NUM = ratio
    X_haphazard, mask = mask_types(X, MASK_NUM, seed=1)
    return X_haphazard, mask

def capricious_(X):
    MASK_NUM = 1
    X_haphazard, mask = mask_types_old(X, MASK_NUM, seed=1)
    return X_haphazard, mask

def load(data_name, ratio):
    file = './data/' + data_name + '.mat'
    mat_data = loadmat(file)
    data = mat_data['data']
    X = data[:, 1:data.shape[1]]
    X = X.astype(float)
    if ratio == 0.5:
        X_haphazard, mask = capricious_(X)
    else:
        X_haphazard, mask = capricious(X, ratio)
    Y = data[:, 0]
    Y = Y.reshape(-1, 1)
    Y[np.where(Y == -1)] = 0
    Y = Y.astype(int)

    return X, Y, X_haphazard, mask