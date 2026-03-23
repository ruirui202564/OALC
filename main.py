from OALC import utils, data_load
from run_OALC import run_OALC
import numpy as np

data_name = 'phishing'
r = 0.5
X, Y, X_haphazard, mask = data_load.load(data_name, r)

# param setting
params = {'budget': 0.2,
         'init_instance': 20,
         'random_thres': 0.1,
         'uncertain_step': 0.01,
         'uncertain_eachClass_thres': 0.9}

# run
n = X.shape[0]
permutations = utils.random_perm_n(n)
bAccuracy = []
time = []
label_cost = []
random_num = []
predict_uncert_num = []
for j in range(10):
    print('permutation:',j+1)
    perm = permutations[j]
    X_perm = X[perm]
    Y_perm = Y[perm]
    X_haphazard_perm = X_haphazard[perm]
    mask_perm = mask[perm]

    result = run_OALC(X_perm, Y_perm, X_haphazard_perm,
                        mask_perm, params)

    bAcc = result['bAccuracy']
    bAccuracy.append(bAcc)
    print(f'{j+1}th permutation: balanced accuracy is {bAcc}')

bAccuracy_mean = np.mean(bAccuracy)
print(f'The average balanced accuracy is {bAccuracy_mean}')

