"""
File: adc.py
Author: Tim Hilt
Date: 2020-10-19
"""

import numpy as np
from numpy.linalg import norm
from sklearn.cluster import KMeans


class ADC:
    def __init__(self, m=16, bs=8):
        self.kmeans = None
        self.database = None
        self.m = m
        self.bs = bs
        self.k = 2**bs

    def _train(self, X):
        self.kmeans = []
        n, d = X.shape
        for i in range(self.m):
            self.kmeans.append(KMeans(self.k)
                               .fit(X[:, i * (d / self.m):(i + 1) * (d / self.m)]))  # Subsequently train Kmeans-models
        self.database = self.transform(X)

    def fit(self, X):
        """Fit KMeans-models to a database-matrix

        Parameters
        ----------
        X : array, shape(n, d)
            Initial non-quantized vectors

        Returns
        -------
        self : ADC()
            Returns an instance of itself
        """
        self._train(X)
        return self

    def predict(self, X):
        return np.argmax(self.predict_proba(X))

    def predict_proba(self, X):
        n, d = X.shape

        lut = np.zeros((len(self.database), self.m))
        # TODO: Could this be vectorized, so that i wouldn't have to loop over both dimensions?
        for i in range(lut.shape[0]):
            for j in range(lut.shape[1]):
                dist = X[j * (d / self.m):(j + 1) * (d / self.m)] - self.kmeans[j].cluster_centers_[self.database[i, j]]
                lut[i, j] = dist @ dist

        scores = lut.sum(axis=1)
        return scores

    def transform(self, X):
        n, d = X.shape
        tmp = []
        for i in range(self.m):
            # uint8 is ok for everything that's smaller than 256 values; maybe there is a better way
            tmp.append(self.kmeans[i].predict(X[:, i * (d / self.m):(i + 1) * (d / self.m)]).astype(np.uint8))
        return np.hstack(tmp)

    def fit_transform(self, X):
        self._train(X)
        return self.database

    def __repr__(self):
        return f"ADC(m={self.m}, bs={self.bs})"

    def __str__(self):
        return f"ADC(m={self.m}, bs={self.bs})"
