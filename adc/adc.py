"""
File: adc.py
Author: Tim Hilt
Date: 2020-10-19
"""

import numpy as np
from sklearn.cluster import KMeans
import progressbar as pb


class ADC:
    """Asymmetric Distance Computation (ADC)

    This class provides an implementation of Asymmetric Distance Computation (ADC)
    originally described in [1]_ and [2]_. The class is able to quantize a high-dimensional
    vector to a sequence of bytes, each having its own KMeans-model.

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

    References
    ----------
    .. [1] Jegou, H., Douze, M., & Schmid, C. (2010). Product quantization
           for nearest neighbor search. IEEE transactions on pattern analysis
           and machine intelligence, 33(1), 117-128.

    .. [2] Jegou, H., Perronnin, F., Douze, M., Sánchez, J., Perez, P., & Schmid,
           C. (2011). Aggregating local image descriptors into compact codes. IEEE
           transactions on pattern analysis and machine intelligence, 34(9), 1704-1716.
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
        for i in pb.progressbar(range(self.m)):
            self.kmeans.append(KMeans(self.k)
                               .fit(X[:, int(i * (d / self.m)):int((i + 1) * (d / self.m))]))

        self.centers = np.zeros((self.kmeans[0].cluster_centers_.shape[0],
                                 self.kmeans[0].cluster_centers_.shape[1],
                                 len(self.kmeans)))
        for i in range(self.m):
            self.centers[..., i] = self.kmeans[i].cluster_centers_
        self.centers = self.centers.transpose((2, 0, 1))

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
        x = X.reshape((1, self.m, -1))

        # TODO: This part needs to be vectorized!
        tmp = []
        for j in range(self.m):
            tmp.append(self.centers[j][self.database[:, j]][:, None, :])  # Introduce new axis in order to be able...
        tmp = np.hstack(tmp)  # ...to stack subtensors horizontally

        # Einstein-summation: Reduce 2nd axis
        scores = np.einsum("ijk,ijk->ij", x-tmp, x-tmp).sum(axis=1)  # Equiv. to squared-norm-distance along 2nd axis
        return scores

    def transform(self, X):
        """Quantize a matrix X to binary codes

        Parameters
        ----------
        X : array, shape(n, d)
            Matrix of global descriptors to be quantized

        Returns
        -------
        np.hstack(tmp) : array, shape(n, m)
            Quantized version of X
        """
        n, d = X.shape
        tmp = []
        for i in range(self.m):
            # uint8 is ok for everything that's smaller than 256 values; maybe there is a better way
            tmp.append(self.kmeans[i].predict(X[:, int(i * (d / self.m)):int((i + 1) * (d / self.m))])
                       .astype(np.uint8)[:, None])
        return np.hstack(tmp)

    def fit_transform(self, X):
        """Subsequently fit the KMeans-models and quantize X

        Parameters
        ----------
        X : array, shape (n, d)
            Matrix of global descriptors to be quantized

        Returns
        -------
        database : array, shape (n, m)
            Quantized matrix X
        """
        self._train(X)
        return self.database

    def __repr__(self):
        return f"ADC(m={self.m}, bs={self.bs})"

    def __str__(self):
        return f"ADC(m={self.m}, bs={self.bs})"
