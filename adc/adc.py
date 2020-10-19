"""
File: adc.py
Author: Tim Hilt
Date: 2020-10-19
"""

import numpy as np
from sklearn.cluster import KMeans


class ADC:
    """
    Parameters
    ----------
    m : int, default=16
        How many subvectors to create
    bs : int, default = 8
        How many bits to encode per subvector

    Attributes
    ----------
    kmeans : list(KMeans)
        List of m KMeans models that fit the m subvectors respectively
    database : array(unit8), shape (n, m)
        Database of quantized vectors
    """
    def __init__(self, m=16, bs=8):
        self.kmeans = None
        self.database = None
        self.m = m
        self.bs = bs
        self.k = 2**bs

    def _train(self, X):
        """Internal utility function that trains the KMeans clusters

        Parameters
        ----------
        X : array, shape (n, d)
            Array of vectors to be quantized

        Returns
        -------
        None : void-function
        """
        self.kmeans = []
        n, d = X.shape
        for i in range(self.m):
            self.kmeans.append(KMeans(self.k)
                               .fit(X[:, int(i * (d / self.m)):int((i + 1) * (d / self.m))]))  # Subsequently train Kmeans-models
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
        """Make prediction using quantized database

        Parameters
        ----------
        X : array, shape (d,)
            Vector for which the nearest neighbor should be searched

        Returns
        -------
        argmin(preds) : int
            The database-index with the minimum distance
        """
        return np.argmin(self.predict_proba(X))

    def predict_proba(self, X):
        """

        Parameters
        ----------
        X : array, shape (d,)
            Vector for which the nearest neighbor should be searched

        Returns
        -------
        scores : array, shape(n,)
            Vector of distances for each database-entry
        """
        d = len(X)

        lut = np.zeros((len(self.database), self.m))
        # TODO: Could this be vectorized, so that i wouldn't have to loop over both dimensions? (Maybe with strides)
        for i in range(lut.shape[0]):
            for j in range(lut.shape[1]):
                dist = X[int(j * (d / self.m)):int((j + 1) * (d / self.m))] \
                       - self.kmeans[j].cluster_centers_[self.database[i, j]]
                lut[i, j] = dist @ dist  # Construct look-up-table

        scores = lut.sum(axis=1)
        return scores

    def transform(self, X):
        n, d = X.shape
        tmp = []
        for i in range(self.m):
            # uint8 is ok for everything that's smaller than 256 values; maybe there is a better way
            tmp.append(self.kmeans[i].predict(X[:, int(i * (d / self.m)):int((i + 1) * (d / self.m))])
                       .astype(np.uint8)[:, None])
        return np.hstack(tmp)

    def fit_transform(self, X):
        self._train(X)
        return self.database

    def __repr__(self):
        return f"ADC(m={self.m}, bs={self.bs})"

    def __str__(self):
        return f"ADC(m={self.m}, bs={self.bs})"
