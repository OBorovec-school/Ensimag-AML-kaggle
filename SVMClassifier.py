import numpy as np
import scipy


class NotFittedException(Exception):
    pass


class SVMClassifier:
    def __init__(self, reg_alpha=0.25,
                 rate=1,
                 max_iter=2000, 
                 min_change=1.0e-6):
        self.fitted = False
        self.w = None
        self.b = 0
        self.reg_alpha = reg_alpha
        self.rate = rate
        self.max_iter = max_iter
        self.min_change = min_change

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        errors = [np.Inf]
        for i in range(1, self.max_iter):
            rate = 1. / (self.rate * i)
            dw, db = self.__grad(X, y)
            self.w -= rate * dw
            self.b -= rate * db
            print(str(self.w) + ' - ' + str(self.b))
            errors.append(self.__error(X, y))
            if np.abs(errors[-1] - errors[-2]) < self.min_change:
                break
        return errors

    def score(self, X, y):
        if not self.fitted:
            raise NotFittedException
        raise NotImplemented

    def predict(self, X):
        if not self.fitted:
            raise NotFittedException
        raise NotImplemented

    def decision_function(self, X):
        return self.__predict(X)

    def __predict(self, X):
        return np.sign(self.__project(X))

    def __error(self, X, y):
        loc_w = self.w[:-1]
        return np.mean(self.__hinge_error(X, y)) + self.reg_alpha * np.sum(loc_w.dot(loc_w))

    def __hinge_error(self, X, y):
        return np.maximum(0, 1 - self.__project(X) * y)

    def __project(self, X):
        return X.dot(self.w) + self.b

    def __grad(self, X, y):
        projection = self.__project(X)
        missclass = projection * y < 1
        print('Miss ' + str(np.sum(missclass)))
        if np.sum(missclass) == 0:
            return self.reg_alpha * 2 * self.w, 0
        else:
            de = X[missclass] * -y[missclass][:, np.newaxis]
            de = np.sum(de, axis=0)
            return de + self.reg_alpha * 2 * self.w, -np.sign(np.sum(y[missclass]))

    def __transform_samples(self, X):
        n_samples, n_features = X.shape
        if isinstance(X, np.ndarray) or isinstance(X, np.matrix):
            return np.column_stack((X, np.ones(n_samples)))
        elif isinstance(X, scipy.sparse.csr.csr_matrix):
            return scipy.sparse.hstack((X, np.ones((n_samples, 1))))
        raise NotImplementedError
