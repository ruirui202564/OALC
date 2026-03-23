from OALC_learner import learner
from OALC.Query import query
import numpy as np

class OALC():
    def __init__(self, D, num_y,
                 budget: float = 0.1,
                 init_instance: int = 20,
                 random_thres: float = 0.1,
                 uncertain_eachClass_thres: float = 0.5,
                 uncertain_step: float = 0.01):

        self.classifier = learner(D, epsilon = np.inf, priors = False)
        self.query = query(budget, init_instance, random_thres, uncertain_eachClass_thres, uncertain_step, num_y)

    def handleInstance(self, x, y, ord_indices):
        y_pred, pred_proba = self.classifier.predict(x, ord_indices)

        # query
        label = self.query.hybrid_sample(x, pred_proba, y)
        if label:
            self.classifier.fit(x, y, ord_indices)

        return y_pred
