import numpy as np


class ArrayScaler:
    def __init__(self, as_scalar=False):

        """as_scalar=True implies that each row is scaled the same way"""
        self.as_scalar = as_scalar
        self.mu = None
        self.sigma = None

    def fit(self, array):

        if self.as_scalar is False:
            self.mu = self.calc_means(array)
            self.sigma = self.calc_stds(array)
        else:
            self.mu = np.mean(array)
            self.sigma = np.std(array)

    def transform(self, array):
        return (array - self.mu) / self.sigma

    def inverse_transform(self, array):
        return (array * self.sigma) + self.mu

    @staticmethod
    def calc_means(array):
        return np.mean(array, axis=0)

    @staticmethod
    def calc_stds(array):
        states_std_ = np.std(array, axis=0)

        for ix, val in enumerate(states_std_):
            if val == 0.0:
                states_std_[ix] = 1.0

        return states_std_
