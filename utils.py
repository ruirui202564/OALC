import numpy as np
from sklearn.metrics import balanced_accuracy_score


def get_ord_indices(X):
    max_ord=10
    indices = []
    for i, col in enumerate(X.T):
        col_nonan = col[~np.isnan(col)]
        col_unique = np.unique(col_nonan)
        if len(col_unique) <= max_ord:
            indices.append(i)
    return indices

# Balanced Accuracy
def BalancedAccuracy(y_actual, y_pred):
    return balanced_accuracy_score(y_actual, y_pred)*100

def get_all_metrics(Y_true, Y_pred, time_taken):
    balanced_accuracy = BalancedAccuracy(Y_true, Y_pred)

    return {"bAccuracy"     : balanced_accuracy,
            "Time"          : time_taken}

def mask_types(X, mask_num, seed):
    X_masked = np.copy(X)
    mask_indices = []
    mask = np.ones_like(X)
    num_rows = X_masked.shape[0]
    num_cols = X_masked.shape[1]
    for i in range(num_rows):
        np.random.seed(seed*num_rows-i)
        num_mask = int(mask_num * num_cols)
        rand_idx = np.random.choice(num_cols, num_mask, replace=False)
        for idx in rand_idx:
            X_masked[i, idx] = np.nan
            mask[i, idx] = 0
            mask_indices.append((i, idx))
    return X_masked, mask

def mask_types_old(X, mask_num, seed):
    X_masked = np.copy(X)
    mask_indices = []
    mask = np.ones_like(X)
    num_rows = X_masked.shape[0]
    num_cols = X_masked.shape[1]
    for i in range(num_rows):
        np.random.seed(seed*num_rows-i)
        for j in range(num_cols//2):
            rand_idx=np.random.choice(2,mask_num,False)
            for idx in rand_idx:
                X_masked[i,idx+2*j] = 0
                mask[i,idx+2*j] = 0
                mask_indices.append((i, idx+2*j))
    return X_masked, mask

def random_perm_n(n):
    np.random.seed(1)
    permutations = []
    for _ in range(10):
        perm = np.arange(n)
        np.random.shuffle(perm)
        permutations.append(perm)

    return permutations