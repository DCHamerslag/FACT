from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals


import numpy as np
from sklearn import metrics
import scipy.sparse as sparse
import data_utils as data


def remove_quantile(X, Y, dists, frac_to_remove):
    """
    Removes the frac_to_remove points from X and Y with the highest value in dists.
    This works separately for each class.
    """
    if len(dists.shape) == 2: # Accept column vectors but reshape
        assert dists.shape[1] == 1
        dists = np.reshape(dists, -1)

    assert len(dists.shape) == 1
    assert X.shape[0] == Y.shape[0]
    assert X.shape[0] == len(dists)
    assert 0 <= frac_to_remove
    assert frac_to_remove <= 1

    frac_to_keep = 1.0 - frac_to_remove
    num_removed_by_class = {}

    idx_to_keep = []
    for y in set(Y):
        num_to_keep = int(np.round(frac_to_keep * np.sum(Y == y)))
        num_removed_by_class[str(y)] = int(np.round(np.sum(Y == y))) - num_to_keep

        idx_to_keep.append(
            np.where(Y == y)[0][np.argsort(dists[Y == y])[:num_to_keep]])

    idx_to_keep = np.concatenate(idx_to_keep)

    X_def = X[idx_to_keep, :]
    Y_def = Y[idx_to_keep]

    return X_def, Y_def, idx_to_keep, num_removed_by_class


def compute_dists_under_Q(
    X, Y,
    Q,
    subtract_from_l2=False, #If this is true, computes ||x - mu|| - ||Q(x - mu)||
    centroids=None,
    class_map=None,
    norm=2):
    """
    Computes ||Q(x - mu)|| in the corresponding norm.
    Returns a vector of length num_examples (X.shape[0]).
    If centroids is not specified, calculate it from the data.
    If Q has dimension 3, then each class gets its own Q.
    """
    if (centroids is not None) or (class_map is not None):
        assert (centroids is not None) and (class_map is not None)
    if subtract_from_l2:
        assert Q is not None
    if Q is not None and len(Q.shape) == 3:
        assert class_map is not None
        assert Q.shape[0] == len(class_map)

    if norm == 1:
        metric = 'manhattan'
    elif norm == 2:
        metric = 'euclidean'
    else:
        raise ValueError('norm must be 1 or 2')

    Q_dists = np.zeros(X.shape[0])
    if subtract_from_l2:
        L2_dists = np.zeros(X.shape[0])

    for y in set(Y):
        if centroids is not None:
            mu = centroids[class_map[y], :]
        else:
            mu = np.mean(X[Y == y, :], axis=0)
        mu = mu.reshape(1, -1)

        if Q is None:   # assume Q = identity
            Q_dists[Y == y] = metrics.pairwise.pairwise_distances(
                X[Y == y, :],
                mu,
                metric=metric).reshape(-1)

        else:
            if len(Q.shape) == 3:
                current_Q = Q[class_map[y], ...]
            else:
                current_Q = Q

            if sparse.issparse(X):
                XQ = X[Y == y, :].dot(current_Q.T)
            else:
                XQ = current_Q.dot(X[Y == y, :].T).T
            muQ = current_Q.dot(mu.T).T

            Q_dists[Y == y] = metrics.pairwise.pairwise_distances(
                XQ,
                muQ,
                metric=metric).reshape(-1)

            if subtract_from_l2:
                L2_dists[Y == y] = metrics.pairwise.pairwise_distances(
                    X[Y == y, :],
                    mu,
                    metric=metric).reshape(-1)
                Q_dists[Y == y] = np.sqrt(np.square(L2_dists[Y == y]) - np.square(Q_dists[Y == y]))

    return Q_dists


def find_feasible_label_flips_in_sphere(X, Y, percentile):
    class_map, centroids, centroid_vec, sphere_radii, slab_radii = data.get_data_params(
        X,
        Y,
        percentile=percentile)

    sphere_dists_flip = compute_dists_under_Q(
        X, -Y,
        Q=None,
        subtract_from_l2=False,
        centroids=centroids,
        class_map=class_map,
        norm=2)

    feasible_flipped_mask = np.zeros(X.shape[0], dtype=bool)

    for y in set(Y):
        class_idx_flip = class_map[-y]
        sphere_radius_flip = sphere_radii[class_idx_flip]

        feasible_flipped_mask[Y == y] = (sphere_dists_flip[Y == y] <= sphere_radius_flip)

    return feasible_flipped_mask


class DataDef(object):
    def __init__(self, X_modified, Y_modified, X_test, Y_test, idx_train, idx_poison):
        self.X_modified = X_modified
        self.Y_modified = Y_modified
        self.X_test = X_test
        self.Y_test = Y_test
        self.idx_train = idx_train
        self.idx_poison = idx_poison

        self.X_train = X_modified[idx_train, :]
        self.Y_train = Y_modified[idx_train]
        self.X_poison = X_modified[idx_poison, :]
        self.Y_poison = Y_modified[idx_poison]

        self.class_map = data.get_class_map()
        self.emp_centroids = data.get_centroids(self.X_modified, self.Y_modified, self.class_map)
        self.true_centroids = data.get_centroids(self.X_train, self.Y_train, self.class_map)
        self.emp_centroid_vec = data.get_centroid_vec(self.emp_centroids)
        self.true_centroid_vec = data.get_centroid_vec(self.true_centroids)

        # Fraction of bad data / good data (so in total, there's 1+epsilon * good data )
        self.epsilon = self.X_poison.shape[0] / self.X_train.shape[0]

    def compute_dists_under_Q_over_dataset(
        self,
        Q,
        subtract_from_l2=False, #If this is true, plots ||x - mu|| - ||Q(x - mu)||
        use_emp_centroids=False,
        norm=2):

        if use_emp_centroids:
            centroids = self.emp_centroids
        else:
            centroids = self.true_centroids

        dists = compute_dists_under_Q(
            self.X_modified, self.Y_modified,
            Q,
            subtract_from_l2=subtract_from_l2,
            centroids=centroids,
            class_map=self.class_map,
            norm=norm)

        return dists

    def get_losses(self, w, b):
        # This removes the max term from the hinge, so you can get negative loss if it's fit well
        losses = 1 - self.Y_modified * (self.X_modified.dot(w) + b)
        return losses