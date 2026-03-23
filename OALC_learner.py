from math import log, e, pi
import numpy as np
from collections import defaultdict

class learner(object):
    def __init__(self, D, epsilon, priors=False):
        self.epsilon = epsilon
        self.D_ = np.int32(D)
        assert self.D_ > 0
        self.priors_ = priors

        # continuous
        self.n_ = {}
        self.mean_ = {}
        self.M2_ = {}
        self.var_ = {}

        # discrete
        self.discrete_counts_ = {}
        self.discrete_n_ = {}

        self.total_ = np.int64(0)
        self.class_counts_ = defaultdict(int)
        self.min_var_ = np.inf

    def fit(self, x, c, ord_indices):
        assert len(x) == self.D_
        self.total_ += 1
        self.class_counts_[c] += 1

        if c not in self.n_:
            D = self.D_
            self.n_[c] = np.ndarray(D, dtype=np.float64, buffer=np.zeros(D))
            self.mean_[c] = np.ndarray(D, dtype=np.float64, buffer=np.zeros(D))
            self.M2_[c] = np.ndarray(D, dtype=np.float64, buffer=np.zeros(D))
            self.var_[c] = np.ndarray(D, dtype=np.float64, buffer=np.zeros(D))

            self.discrete_counts_[c] = {}
            self.discrete_n_[c] = np.zeros(D, dtype=np.int64)

            for i in ord_indices:
                self.discrete_counts_[c][i] = defaultdict(int)

        for i in range(self.D_):
            if x[i] is None:
                continue

            if i in ord_indices:
                self._update_discrete_feature(x[i], c, i)
            else:
                self._update_continuous_feature(x[i], c, i)

    def _update_discrete_feature(self, value, c, i):
        self.discrete_counts_[c][i][value] += 1
        self.discrete_n_[c][i] += 1

    def _update_continuous_feature(self, value, c, i):
        self.n_[c][i] += 1
        n = self.n_[c][i]
        delta = value - self.mean_[c][i]
        self.mean_[c][i] += delta / n
        self.M2_[c][i] += delta * (value - self.mean_[c][i])

        if n < 2:
            self.var_[c][i] = np.float64(0)
        else:
            self.var_[c][i] = self.M2_[c][i] / (n - 1)

        if self.var_[c][i] < self.min_var_ and self.var_[c][i] != 0:
            self.min_var_ = self.var_[c][i]

    def _pdf_continuous(self, x, m, v):
        v += self.min_var_ / 1000 if self.min_var_ != np.inf else 1e-9
        return (-0.5 * log(2 * pi * v) - ((x - m) ** 2 / (2 * v)))

    def _prob_discrete(self, value, c, feature_idx):
        if c not in self.discrete_counts_ or feature_idx not in self.discrete_counts_[c]:
            return log(self.epsilon)

        value_count = self.discrete_counts_[c][feature_idx][value]
        total_count = self.discrete_n_[c][feature_idx]

        if total_count == 0:
            return log(self.epsilon)

        unique_values = len(self.discrete_counts_[c][feature_idx])
        k = max(unique_values, 1)

        alpha = 1.0
        smoothed_prob = (value_count + alpha) / (total_count + alpha * k)

        return log(max(smoothed_prob, 1e-10))

    def _get_class_prior(self, c):
        if self.priors_ and self.total_ > 0:
            return log(self.class_counts_[c] / self.total_)
        else:
            return 0.0

    def predict(self, x, ord_indices):
        if not self.mean_ and not self.discrete_counts_:
            return 0, [1.0]

        scores = {}

        all_classes = set()
        if self.mean_:
            all_classes.update(self.mean_.keys())
        if self.discrete_counts_:
            all_classes.update(self.discrete_counts_.keys())

        for c in all_classes:
            score = self._get_class_prior(c)

            for i in range(self.D_):
                if x[i] is None or np.isnan(x[i]):
                    continue

                if i in ord_indices:
                    score += self._prob_discrete(x[i], c, i)
                else:
                    if (c in self.mean_ and self.n_[c][i] > 0):
                        score += self._pdf_continuous(x[i], self.mean_[c][i], self.var_[c][i])
                    else:
                        score += log(self.epsilon)

            scores[c] = score

        if not scores:
            return 0, [1.0]

        classes = list(scores.keys())
        score_values = np.array(list(scores.values()))

        if not np.any(np.isfinite(score_values)):
            default_probs = np.ones(len(classes)) / len(classes)
            prob_dict = {c: p for c, p in zip(classes, default_probs)}
            sorted_classes = sorted(prob_dict.keys())
            prob_list = [prob_dict[c] for c in sorted_classes]
            return classes[0], prob_list

        max_score = np.max(score_values)
        if not np.isfinite(max_score):
            default_probs = np.ones(len(classes)) / len(classes)
            prob_dict = {c: p for c, p in zip(classes, default_probs)}
            sorted_classes = sorted(prob_dict.keys())
            prob_list = [prob_dict[c] for c in sorted_classes]
            return classes[0], prob_list

        score_diff = score_values - max_score
        score_diff[~np.isfinite(score_diff)] = -700

        exp_scores = np.zeros_like(score_values)
        valid_mask = score_diff > -700
        exp_scores[valid_mask] = np.exp(score_diff[valid_mask])

        total_exp = np.sum(exp_scores)
        if total_exp == 0:
            probabilities = np.ones_like(score_values) / len(score_values)
        else:
            probabilities = exp_scores / total_exp

        prob_dict = {c: p for c, p in zip(classes, probabilities)}
        sorted_classes = sorted(prob_dict.keys())
        prob_list = [prob_dict[c] for c in sorted_classes]

        best_class = classes[np.argmax(probabilities)]

        return best_class, prob_list