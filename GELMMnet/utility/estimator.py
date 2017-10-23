""""
Copyright (c) 2015, Selective Inference development team
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials provided
       with the distribution.

    * The names of any contributors to this software
       may not be used to endorse or promote products derived
       from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

import numpy as np
from numba import jit

from GELMMnet.utility.inference import _cv_grid_search


@jit(nopython=True, parallel=True)
def _sandwich_estimator(X, y, l2, P, beta, active, inactive, B):
    """
        Bootstrap estimator of covariance of

    .. math::

        (\bar{\beta}_E, X_{-E}^T(y-X_E\bar{\beta}_E)

    the ridge regression estimator coefficients and inactive correlation with the ridge regression residuals.

    :param X: regressors
    :param y: regressies
    :param P: ridge penalty matrix
    :param beta: the regression coefficient that are not 0
    :param active: indices of features with coefficiance != 0
    :param inactive: boolean list of features with coefficiens == 0
    :param B: bootstrap iterations
    :return:
    """
    n,m = X.shape
    beta_full = np.zeros(m)
    beta_full[active] = beta

    # make sure active / inactive are bool

    active_bool = np.zeros(m, np.bool)
    active_bool[active] = 1

    inactive_bool = np.zeros(m, np.bool)
    inactive_bool[inactive] = 1


    nactive = active_bool.sum()
    first_moment = np.zeros(m)
    second_moment = np.zeros((m, nactive))

    X_full, y, beta_overall, Qinv, I, nactive, ntotal = _prepare_pairs_boostrap(X, y, P, l2, active_bool)
    sqrt_scale = np.sqrt(1.0)

    for b in range(B):
        indices = np.random.choice(n, n, replace=True)
        Z_star = _pairs_bootstrap(X_full[indices], y[indices], Qinv, I, nactive, ntotal, indices, sqrt_scale)
        first_moment += Z_star
        second_moment += np.outer(Z_star, Z_star[:nactive])

    first_moment /= B
    second_moment /= B

    cov = second_moment - np.outer(first_moment, first_moment[:nactive])

    return cov


@jit(nopython=True)
def _restricted_Mest(X, y, l2, P, active):
    """
    fitting generalized ridge regression

    b_E = (X.T * X + l2*P)^-1 * X.T * y

    :param active:
    :param solver_args:
    :return:
    """
    Xe = X[:, active]
    Q = np.dot(Xe.T, Xe) + l2*P
    Qinv = np.linalg.inv(Q)

    return np.dot(Qinv, np.dot(Xe.T, y))


@jit(nopython=True)
def _prepare_pairs_boostrap(X, y, P, l2, active, beta_full=None, inactive=None):
    """
    precalculation needed for pairs boostrapping

    :param X: input matrix
    :param y: regress
    """
    if beta_full is None:
        beta_active = _restricted_Mest(X, y, P, l2, active)
        beta_full = np.zeros(X.shape[1])
        beta_full[active] = beta_active
    else:
        beta_active = beta_full[active]

    X_active = X[:, active]
    nactive = active.sum()
    ntotal = nactive
    if inactive is not None:
        X_inactive = X[:, inactive]
        ntotal += inactive.sum()

    # on step estimator for ridge regression
    Q = np.dot(X_active.T, X_active) + l2*P
    Qinv = np.linalg.inv(Q)
    if inactive is not None:
        I = np.dot(X_inactive.T, Qinv)
    else:
        I = None

    if inactive is not None:
        X_full = np.hstack([X_active, X_inactive])
        beta_overall = np.zeros(X_full.shape[1])
        beta_overall[:nactive] = beta_active
    else:
        X_full = X_active
        beta_overall = beta_active

    return X_full, y, beta_overall, Qinv, I, nactive, ntotal


@jit(nopython=True, cache=True)
def _pairs_bootstrap(X_star, y_star, beta_overall, Qinv, I, nactive, ntotal,  sqrt_scaling):
    """
    pairs bootstrap of (beta_hat_active, -grad_inactive(beta_hat_active))
    """
    score = np.dot(X_star.T, (y_star - np.dot(X_star, beta_overall)))
    result = np.zeros(ntotal)
    result[:nactive] = np.dot(Qinv, score[:nactive])

    if ntotal > nactive:
        result[nactive:] = score[nactive:] - np.dot(I, score[:nactive])

    result[:nactive] *= sqrt_scaling
    result[nactive:] /= sqrt_scaling
    return result


@jit(nopython=True, catch=True)
def _boostrap_CVerror_curve(Xt, Xv, yt, yv, P, delta, param_grid, isIntercept, scale=0.01):
        """
        bootstraps the CV curve

        :param scale:
        :return:
        """
        l1, l2, error, error_curve = _cv_grid_search(Xt, Xv, yt, yv, P, delta, isIntercept,
                                                    scale, param_grid, 5, 100, 1e-5)
        return error_curve


@jit(nopython=True, parallel=True)
def _nonparametric_cov_bootstrap(Xt, Xv, yt, yv, l2, P, delta, param_grid, isIntercept, active, B, cross_term=True, scale=0.01):
        """
        returns estimates of covariance matrices: boot_target with itself,
        and the blocks of (boot_target, boot_other) for other in cross_terms

        :param Xt: possibly rotated matrix input matrix
        :param Xv: not rotated input matrix
        :param yt: possibly rotated output vector
        :param yv: not rotated output vector
        :param l2: l2 regularization hyperparameter
        :param P: l2 penalty matrix
        :param delta:
        :param param_grid: grid-search parameter grid
        :param active: active indices after elastic net grid search optimization
        :param B: number of bootstrap samples
        :param cross_term: if a cross term between CV covariance and model selection covariance should be calculated
        :param scale: scale of the random noise that is added to the CV error
        :return:
        """

        # setup variables
        n, m = Xt.shape
        mean_target = 0
        if cross_term is not None:
            mean_cross = 0
            outer_cross = 0
        outer_target = 0

        # setup pairs_bootstrapper
        active_bool = np.zeros(m, np.bool)
        active_bool[active] = 1
        X_full, y, beta_overall, Qinv, I, nactive, ntotal = _prepare_pairs_boostrap(Xt, yt, P, l2, active_bool)
        sqrt_scale = np.sqrt(1.0)

        # start bootstrap
        for j in range(B):
            indices = np.random.choice(n, size=(n,), replace=True)
            boot_target = _pairs_bootstrap(X_full[indices], y[indices], Qinv, I,
                                           nactive, ntotal, indices, sqrt_scale)[:active_bool.sum()]
            mean_target += boot_target
            outer_target += np.outer(boot_target, boot_target)

            # bootstrap CV curve
            if cross_term:
                boot_sample = _boostrap_CVerror_curve(Xt[indices], Xv[indices], yt[indices], yv[indices],
                                                      P, delta, param_grid, isIntercept, scale)
                mean_cross += boot_sample
                outer_cross += np.outer(boot_target, boot_sample)

        mean_target /= B
        outer_target /= B

        if cross_term:
            mean_cross /= B
            outer_cross /= B

        cov_target = outer_target - np.outer(mean_target, mean_target)
        if not cross_term:
            return cov_target
        return [cov_target] + [_o - np.outer(mean_target, _m) for _m, _o in zip(mean_cross, outer_cross)]















