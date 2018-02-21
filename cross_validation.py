import numpy as np


class CrossValidate:

    def __init__(self, nb_samples, nb_folds, rand_seed=None):
        assert nb_samples > nb_folds, 'too few samples'
        self.nb_samples = nb_samples
        self.nb_folds = nb_folds

        self.set_indexes = list(range(self.nb_samples))

        if rand_seed is not False:
            np.random.seed(rand_seed)
            np.random.shuffle(self.set_indexes)

        self.iter = 0

    def __iter__(self):
        step = int(np.ceil(self.nb_samples / float(self.nb_folds)))
        for i in range(0, self.nb_samples, step):
            inds_test = self.set_indexes[i:i + step]
            inds_train = self.set_indexes[:i] + self.set_indexes[i + step :]
            yield inds_train, inds_test

    def __len__(self):
        return int(self.nb_folds)


def compute_cross_vals_score(model, X, y, nb_folds=10):
    cv = CrossValidate(len(y), nb_folds)
    list_scores = []
    for inds_train, inds_test in cv:
        model.fit(X[inds_train, :],  y[inds_train])
        score = model.score(X[inds_test, :], y[inds_test])
        list_scores.append(score)
    return np.mean(list_scores)

