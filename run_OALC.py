from OALC.utils import get_ord_indices, get_all_metrics
from OALC.OALC_ import OALC
from tqdm import tqdm
import numpy as np
import time

def run_OALC(X, Y, X_haphazard, mask, params):
    X_masked = np.ones_like(X) * np.nan
    X_masked[mask.astype(bool)] = X[mask.astype(bool)]
    all_ord_indices = get_ord_indices(X_masked)
    num_classes = len(np.unique(Y))

    data = []
    for i in range(0, len(Y)):
        x, x_mask, y = X_haphazard[i], mask[i], Y[i]
        X_dict = []
        d_dict = []
        for j in range(0, len(x)):
            if x_mask[j] == 1:
                X_dict.append(x[j])
            elif x_mask[j] == 0:
                X_dict.append(None)
        d_dict.append(X_dict)
        d_dict.append(float(y[0]))
        data.append(d_dict)

    Y_pred = []
    start_time = time.time()
    c = []

    for k, g in data:
        c.append(g)

    model = OALC(len(data[0][0]), num_classes, params['budget'], params['init_instance'], params['random_thres'],
                 params['uncertain_eachClass_thres'], params['uncertain_step'])
    for r, t in tqdm(zip(data, c)):
        X = r[0]
        y = r[1]
        y_pred = model.handleInstance(X, y, all_ord_indices)
        Y_pred.append(y_pred)

    taken_time = time.time() - start_time

    metrics = get_all_metrics(Y, np.array(Y_pred).reshape(-1, 1), taken_time)
    metrics["label_cost"] = model.query.cost
    metrics["random"] = model.query.random
    metrics["predict_uncertain"] = model.query.predict_uncertain

    return metrics