import numpy as np
from BaseQuery import BaseQuery

class query(BaseQuery):
    def __init__(self,
                 budget: float = 0.1,
                 init_instance: int = 50,
                 random_thres: float = 0.1,
                 uncertain_eachClass_thres: float = 0.9,
                 uncertain_step: float = 0.01,
                 number_of_classes: int = 2,
                 random_seed: int = 42):
        super().__init__(budget, init_instance, random_thres, random_seed)

        self.uncertain_step = uncertain_step
        self.multi_certain_thres = [uncertain_eachClass_thres] * number_of_classes

        self.is_uncertain = False
        self.uncertain_eachClass_thres = uncertain_eachClass_thres

    def uncertainty_eachClass(self, predict, true_label, cost_now, predict_idx):
        label = False
        predict_prob = min(1.0, max(0.0, predict[predict_idx]))
        if predict_prob <= self.multi_certain_thres[predict_idx]:
            label = True
            self.is_uncertain = True
            if true_label == predict_idx:
                self.multi_certain_thres[predict_idx] = self.multi_certain_thres[predict_idx] * (1 - self.uncertain_step)

        return label

    def hybrid_sample(self, x, predict, true_label):  # true_label only can be used after queried
        self.instances_seen += 1
        self.is_uncertain = False

        label = False

        predict = np.array(predict)
        predict_idx = np.argmax(predict)

        if self.instances_seen <= self.init_instance:
            label = True
        else:
            self.update_cost()
            if self.cost <= self.budget:
                # uncertainty
                label_uncertain_selective = self.uncertainty_eachClass(predict, true_label, self.cost, predict_idx)
                if label_uncertain_selective:
                    label = True
                    self.predict_uncertain += 1
                else:
                    # random
                    label_random = self.random_sample_()
                    if label_random:
                        self.random += 1
                        label = True

            if label and predict_idx != true_label and not self.is_uncertain:
                self.multi_certain_thres[predict_idx] = max(self.uncertain_eachClass_thres,
                                                            self.multi_certain_thres[predict_idx] * (1 + self.uncertain_step))

        if label:
            self.labeling_cost += 1

        return label