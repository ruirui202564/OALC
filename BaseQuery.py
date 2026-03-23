import numpy as np
import random

class BaseQuery:
    def __init__(self,
                 budget: float = 0.1,
                 init_instance: int = 50,
                 random_thres: float = 0.1,
                 random_seed: int = 42):
        self.init_instance = init_instance
        self.random_thres = random_thres

        self.budget = budget
        self.labeling_cost = 0
        self.instances_seen = 0
        self.cost = 0
        self.labeled_instances = []

        self.random = 0
        self.predict_uncertain = 0

        if random_seed > 0:
            random.seed(random_seed)
            np.random.seed(random_seed)
            self.classifier_random = random.Random(random_seed)
        else:
            self.classifier_random = random.Random()

    def random_sample(self):
        label = False
        r = self.classifier_random.random()
        if r < self.random_thres:
            label = True
        return label

    def random_sample_(self):
        label = False
        r = self.classifier_random.random()
        if r < self.budget: # epsilon = B
            label = True
        return label

    def update_cost(self):
        self.cost = self.labeling_cost / self.instances_seen if self.instances_seen > 0 else 0
        return self.cost

    def update_labelInstance(self):
        self.labeled_instances.append(self.instances_seen)
