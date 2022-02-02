from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals


import numpy as np
from sklearn import metrics
import scipy.sparse as sparse
import data_utils as data


def compute_dists_under_Q(
    X, Y,
    Q,
    subtract_from_l2=False, # If this is true, computes ||x - mu|| - ||Q(x - mu)||
    centroids=None,
    class_map=None,
    norm=2):
    """
    Computes ||Q(x - mu)|| in the corresponding norm.
    Returns a vector of length num_examples (X.shape[0]).
    If centroids is not specified, calculate it from the data.
    If Q has dimension 3, then each class gets its own Q.
    """
    # Check if the class maps and centroids exist
    if (centroids is not None) or (class_map is not None):
        assert (centroids is not None) and (class_map is not None)
    if subtract_from_l2:
        assert Q is not None
    if Q is not None and len(Q.shape) == 3:
        assert class_map is not None
        assert Q.shape[0] == len(class_map)

    # Check which metric has to be used. By default this is euclidean
    if norm == 1:
        metric = 'manhattan'
    elif norm == 2:
        metric = 'euclidean'
    else:
        raise ValueError('norm must be 1 or 2')

    # If this is true, computes ||x - mu|| - ||Q(x - mu)||
    Q_dists = np.zeros(X.shape[0])
    if subtract_from_l2:
        L2_dists = np.zeros(X.shape[0])
    
    for y in set(Y):
        # If centroids is not specified, calculate it from the data.
        if centroids is not None:
            mu = centroids[class_map[y], :]
        else:
            mu = np.mean(X[Y == y, :], axis=0)
        mu = mu.reshape(1, -1)

        if Q is None:   # Assume Q = identity
            Q_dists[Y == y] = metrics.pairwise.pairwise_distances(
                X[Y == y, :],
                mu,
                metric=metric).reshape(-1)

        else:
            # If Q has dimension 3, then each class gets its own Q.
            if len(Q.shape) == 3:
                current_Q = Q[class_map[y], ...]
            else:
                current_Q = Q

            # Handle sparse matrices
            if sparse.issparse(X):
                XQ = X[Y == y, :].dot(current_Q.T)
            else:
                XQ = current_Q.dot(X[Y == y, :].T).T
            muQ = current_Q.dot(mu.T).T

            # Calculate pairwise distances (sklearn)
            Q_dists[Y == y] = metrics.pairwise.pairwise_distances(
                XQ,
                muQ,
                metric=metric).reshape(-1)

            # If this is true, computes ||x - mu|| - ||Q(x - mu)||
            if subtract_from_l2:
                L2_dists[Y == y] = metrics.pairwise.pairwise_distances(
                    X[Y == y, :],
                    mu,
                    metric=metric).reshape(-1)
                Q_dists[Y == y] = np.sqrt(np.square(L2_dists[Y == y]) - np.square(Q_dists[Y == y]))

    # Returns a vector of length num_examples (X.shape[0]).
    return Q_dists


