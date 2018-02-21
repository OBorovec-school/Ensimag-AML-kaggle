import numpy as np
import scipy
import pandas as pd
from scipy import sparse


class NotFittedException(Exception):
    pass


class WrongInputException(Exception):
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

    def fit(self, X, y, X_inv=None):
        X, y, X_inv = self.__input_data_check_transform(X, y, X_inv)
        n_samples, n_features = X.shape
        if X_inv is None:
            X_inv = self.__get_X_inv(X, y)
        self.w = np.zeros(n_features)
        errors = [np.Inf]
        for i in range(1, self.max_iter):
            rate = 1. / (self.rate * i)
            dw, db = self.__grad(X, y, X_inv)
            self.w -= rate * dw
            self.b -= rate * db
            # print(str(self.w) + ' - ' + str(self.b))
            errors.append(self.__error(X, y))
            if np.abs(errors[-1] - errors[-2]) < self.min_change:
                break
        self.fitted = True
        return errors

    def score(self, X, y):
        if not self.fitted:
            raise NotFittedException
        X, y, _ = self.__input_data_check_transform(X, y)
        predictions = self.__predict(X)
        return np.sum(predictions == y) / len(y)

    def predict(self, X):
        if not self.fitted:
            raise NotFittedException
        X, _, _ = self.__input_data_check_transform(X)
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

    def __grad(self, X, y, X_inv):
        projection = self.__project(X)
        missclass = projection * y < 1
        # print('Miss ' + str(np.sum(missclass)))
        if np.sum(missclass) == 0:
            return self.reg_alpha * 2 * self.w, 0
        else:
            de = np.sum(X_inv[missclass], axis=0) / np.sum(missclass)
            de = de.A1
            return de + self.reg_alpha * 2 * self.w, -np.sign(np.sum(y[missclass]))

    def __get_X_inv(self, X, y):
        n_samples, n_features = X.shape
        X_inv = X.copy()
        for i in range(n_samples):
            X_inv[i, :] = X_inv[i, :] * -int(y[i])
        return X_inv

    def __input_data_check_transform(self, X=None, y=None, X_inv=None):
        # X data
        if X is not None:
            if isinstance(X, sparse.csr_matrix):
                pass
            elif isinstance(X, np.ndarray):
                X = sparse.csr_matrix(X)
            elif isinstance(X, np.matrix):
                X = sparse.csr_matrix(X)
            elif isinstance(X, pd.DataFrame):
                X = sparse.csr_matrix(X.as_matrix())
            else:
                raise WrongInputException
        n_samples, n_features = X.shape

        # y data
        if y is not None:
            if isinstance(y, np.ndarray):
                y = y.flatten()
            elif isinstance(y, np.matrix):
                y = y.A1
            elif isinstance(y, pd.Series):
                y = y.values
            else:
                raise WrongInputException
            if X is not None and len(y) != n_samples:
                raise WrongInputException

        if X_inv is not None  and X is not None \
                and X.shape != X_inv.shape \
                and type(X) != type(X_inv):
            X_inv = None

        return X, y, X_inv
