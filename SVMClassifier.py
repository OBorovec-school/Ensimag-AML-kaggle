import numpy as np

class NotFittedException(Exception):
    pass


class SVMClassifier:
    def __init__(self):
        self.fitted = False
        self.w = None
        self.b = None

    def fit(self, X, y):
        pass

    def score(self, X, y):
        if not self.fitted:
            raise NotFittedException
        pass

    def predict(self, X):
        if not self.fitted:
            raise NotFittedException
        pass

    def __project(self, x):
        return np.sign(np.dot(x, self.w) + self.b)