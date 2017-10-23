from collections import OrderedDict
from itertools import starmap
import os
os.environ["NUMBA_WARNINGS"] = "1"

import numpy as np
from numba import jit, prange

from sklearn.model_selection import KFold

from GELMMnet.utility.kernels import kinship


@jit(nopython=True, cache=True)
def _eval_neg_log_likelihood(ldelta, Uy, S):
    """
    Evaluate the negative log likelihood of a random effects model:
    nLL = 1/2(n_s*log(2pi) + logdet(K) + 1/ss * y^T(K + deltaI)^{-1}y,
    where K = USU^T.

    :param ldelta: log-transformed ratio sigma_gg/sigma_ee
    :param Uy: transformed outcome: n_s x 1
    :param S: eigenvectors of K: n_s
    :return: negative log likelihood value
    """

    n_s = Uy.shape[0]
    delta = np.exp(ldelta)

    # evaluate log determinant
    Sd = S + delta
    ldet = np.sum(np.log(Sd))

    # evaluate the variance
    Sdi = 1.0 / Sd
    Uy = Uy.flatten()
    ss = 1. / n_s * (Uy * Uy * Sdi).sum()
    # evaluate the negative log likelihood
    nLL = 0.5 * (n_s * np.log(2.0 * np.pi) + ldet + n_s + n_s * np.log(ss))

    return nLL


@jit(nopython=True, cache=True)
def _calc_glnet_obj(S, y, Pw, w, l1, l2, n, m):
    """
    Calculates the gelnet objective

     1/(2*n)*Sum(y_i - S)^2 + l1*sum(|w|) +l2/2*w^T*P*w

    :return: gelnet regularized loss function
    """
    loss = np.sum(np.power(y[:, 0] - S, 2.0))
    reg_l1 = np.sum(np.abs(w))
    reg_l2 = np.dot(w.T, Pw)

    return loss / (2.0 * n) + l1 * reg_l1 + 0.5 * l2 * reg_l2


@jit(nopython=True, cache=True)
def _snap_threshold(residual, gamma):
    """
    soft-thresholding function to accelerate lasso regression

    :param residual: residual to should be snapped back to zero
    :param gamma: threshold boundary
    :return: snapped residual
    """
    if np.less(np.fabs(residual), gamma):
        return 0.0
    elif np.less(residual, 0.0):
        return residual + gamma
    else:
        return residual - gamma


@jit(nopython=True, cache=True)
def _update_wj(X, y, P, w, l1, l2, S, Pw, n, m, j):
    """
    Update rule based on coordinate descent
    """
    numerator = np.sum(X[:, j] * (y[:, 0] - S + X[:, j] * w[j]))

    numerator /= n
    numerator -= l2 * (Pw[j] - P[j, j] * w[j])

    # snap value to zero again
    numerator = _snap_threshold(numerator, l1)

    if np.equal(numerator, 0.0):
        return 0.0
    denom = np.sum(np.power(X[:, j], 2))
    denom /= n
    denom += l2 * P[j, j]

    return numerator / denom


#@jit(nopython=True, cache=True)
@jit(nopython=True, parallel=True)
def _optimize_gelnet(y, X, P, l1, l2, S, Pw, n, m, max_iter, eps, w, b, with_intercept):
    """
    Coordinate descent of a generalized elastic net

    :return: weights, intercept
    """
    obj_old = _calc_glnet_obj(S, y, Pw, w, l1, l2, n, m)

    # start optimization
    # optimize for max_iter steps
    for i in range(max_iter):
        # update each weight individually
        #for j in range(m):
        for j in prange(m):
            w_old = w[j]

            w[j] = _update_wj(X, y, P, w, l1, l2, S, Pw, n, m, j)
            wj_dif = w[j] - w_old

            # update fit
            if np.not_equal(wj_dif, 0):
                S += X[:, j] * wj_dif
                Pw += P[:, j] * wj_dif

        # update bias
        b_diff = 0
        if with_intercept:
            old_b = b
            b = np.mean(y[:, 0] - (S - b))
            b_diff = b - old_b

        # update fits accordingly
        S += b_diff

        # calculate objective and test for convergence
        obj = _calc_glnet_obj(S, y, Pw, w, l1, l2, n, m)
        abs_dif = np.fabs(obj - obj_old)

        # optimization converged?
        if np.less(abs_dif, eps):
            break
        else:
            obj_old = obj

    return w, b


@jit(nopython=True, cache=True)
def _predict(X, y, X_tilde, w, b, delta, isLMM):
    """
    predicts the phenotype based on the trained model
    following Rasmussen and Williams 2006

    :return: y_tilde the predicted phenotype
    """
    if isLMM:

        n_test = X_tilde.shape[0]
        n_train = X.shape[0]
        idx = w.nonzero()[0]
        Xtmp = np.concatenate((X, X_tilde), axis=0)

        # calculate Covariance matrices
        K = kinship(Xtmp)
        idx_tt = np.arange(n_train)
        idx_vt = np.arange(n_train, n_test + n_train)

        K_tt = K[idx_tt][:, idx_tt]
        K_vt = K[idx_vt][:, idx_tt]

        if idx.shape[0] == 0:
            return np.dot(K_vt, np.linalg.solve(K_tt + delta * np.eye(n_train), y[:, 0]))

        return np.dot(X_tilde[:, idx], w[idx]) + b + np.dot(K_vt, np.linalg.solve(K_tt + delta * np.eye(n_train),
                                                                                 y[:, 0] - np.dot(X[:, idx],
                                                                                                w[idx]) + b))
    else:
        return np.dot(X_tilde, w) + b


def max_l1(y, X):
    """
    returns the upper limit of l1 (i.e., smallest value that yields a model with all zero weights)

    """
    b = np.mean(y)
    xy = np.mean(X * (y - b), axis=0)
    return np.max(np.fabs(xy))


@jit(nopython=True)
def alpha_grid(l1_ratio, Xy, n_samples,  n_alphas, eps=1e-3):
    """
    Compute the grid of alpha values for elastic net parameter search
    :param X:
    :param y:
    :param l1_ratio:
    :param n_alpha:
    :return:
    """
    if Xy.ndim == 1:
        Xy = Xy[:, np.newaxis]

    alpha_max = (np.sqrt(np.sum(Xy ** 2, axis=1)).max() /
                 (n_samples * l1_ratio))

    if alpha_max <= np.finfo(float).resolution:
        alphas = np.empty(n_alphas)
        alphas.fill(np.finfo(float).resolution)
        return alphas

    return np.logspace(np.log10(alpha_max * eps), np.log10(alpha_max), num=n_alphas)[::-1]


@jit(nopython=True, cache=True)
def _rmse(y, ypred):
    """
    Calculates the root mean squared error

    :return: RMSE
    """
    return np.sqrt(np.nanmean(np.power((y - ypred), 2)))


@jit(nopython=True, cache=True)
def _corr(y, ypred):
    """
    Calculates the pearson correlation coefficient
    NOTE: its negative correlation to be able to use it
    in the CV grid search

    :param y:
    :param ypred:
    :return:
    """
    return -1./len(y) * ((ypred-ypred.mean())*(y-y.mean())).sum()/(ypred.std()*y.std())


@jit(nopython=True, cache=True)
def _parameter_search(fold, l1, l2, delta, isIntercept, ytrain, ypred, Xtrain, Xpred, ytest, Xtest, P, eps, max_iter, n, m, isLMM):
    """
    Function for grid search evaluation

    :return:  fold_idx,error,l1,l2
    """
    w = np.zeros(m)
    b = np.mean(ypred) if isIntercept else 0.0

    # initial parameter estimates
    S = np.dot(Xtrain, w) + b
    Pw = np.dot(P, w)

    w, b = _optimize_gelnet(ytrain, Xtrain, P, l1, l2, S, Pw, n, m,  max_iter, eps, w, b, isIntercept)

    yhat = _predict(Xpred, ypred, Xtest, w, b, delta, isLMM)
    return fold, _rmse(ytest[:, 0], yhat), l1, l2


#@jit(parallel=True)
def _cv_grid_search(Xt, Xv, yt, yv, P, delta, isIntercept, scale, param_grid, nof_cv, max_iter, eps, isLMM):
    """
    TODO: rewrite in pure numpy python as to use jit auto-parallelization

    CV grid search routine

    :param Xt:
    :param Xv:
    :param yt:
    :param yv:
    :param P:
    :param eps:
    :param n:
    :param m:
    :param scale:
    :param param_grid:
    :param nof_cv:
    :param max_iter:
    :return: optimal_idx, optimal_l1, optimal_l2, error, ErrorCurve
    """

    def generate_grid():
        for i, (train_id, test_id) in enumerate(cv.split(Xt)):
            Xtrain, Xtest, Xpred = Xt[train_id], Xv[test_id], Xv[train_id]
            ytrain, ytest, ypred = yt[train_id], yv[test_id], yv[train_id]
            n, m = Xtrain.shape
            for (l1,l2) in param_grid:
                    yield [i, l1, l2, delta, isIntercept,
                           ytrain, ypred, Xtrain, Xpred, ytest, Xtest, P, eps, max_iter, n, m, isLMM]

    cv = KFold(nof_cv)

    grid_result = []

    for p in generate_grid():
        grid_result.append(_parameter_search(*p))

    # summarize grid search results
    sum_res = {}

    for fold, error, l1, l2 in grid_result:
        # add a small error on top of the prediction array (needed for CV corrected post-selection analysis)
        if scale:
            error = error + np.random.normal(0, scale=scale)

        sum_res.setdefault((l1, l2), []).append(error)

    for k, v in sum_res.items():
        sum_res[k] = np.mean(v)

    # find best l1, l2 pair across the folds
    # has to be for looped otherwise it does not work with numba I presume
    idx = -1
    l1,l2 = 0,0
    error = float("inf")
    for i, (param, _error) in enumerate(sum_res.items()):
        if _error < error:
            l1,l2 = param
            idx = i
            error = _error

    return idx, l1, l2, error, sum_res

