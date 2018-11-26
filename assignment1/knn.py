import numpy as np
import numpy
from numpy.linalg import norm

class Knn(object):
    """Naive implementation of KNN

    Attributes:
        Xref(ndarray): flattened matrix
        Yref(ndarray): corresponding labels

    Methods:
        train(X, Y): train based on data X and labels Y
        pred(X): predicts X labels
    """

    def __init__(self, arg1):
        self.arg1 = []
    
    def __init__(self):
        pass

    def train(self, X, Y):
        self.Xref = X
        self.Yref = Y.astype(int)
        self.nclasses = 10

    def compute_dist(self, p1, p2, dist):
        if dist == 'L1': opt = 1
        elif dist == 'L2': opt = 2
        else: raise('Not accepted norm option.')
        diff = p2 - p1

        return norm(diff, ord=opt)

    def predict(self, X, k, dist='L1'):
        nrows, _ = X.shape
        Y = np.ndarray(nrows, dtype=int)

        for r in range(nrows):
            Y[r] = self.predict_single(X[r, :], k, dist)
        return Y

    def predict_single(self, x, k, dist='L1'):
        nrows, ncols = self.Xref.shape
        dist = np.ndarray(self.Yref.shape)
        votes = np.zeros(self.nclasses, dtype=int)

        for r in range(nrows):
            dist[r] = self.compute_dist(x, self.Xref[r, :], 'L2')    # naive

        indices = dist.argsort()
        for idx in indices[:k]:
            votes[self.Yref[idx]] += 1
        return (votes.argmax())
