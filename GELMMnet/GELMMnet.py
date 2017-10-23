"""
GELMMnet.py

Author:		Benjamin Schubert
Year:		2015
Group:		Debora Marks Group
Institutes:	Systems Biology, Harvard Medical School, 200 Longwood Avenue, Boston, 02115 MA, USA

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.


The implementation is based on Barbara Rakitsch's implementation of LMM-Lasso (https://github.com/BorgwardtLab/LMM-Lasso)
, Artem Skolov's implementation of GELnet (https://github.com/cran/gelnet), and Ryan Tibshirani et al.'s implementation
of selectiveInference (https://cran.r-project.org/web/packages/selectiveInference/index.html)

"""
import multiprocessing as mp
from collections import OrderedDict
from itertools import starmap

import itertools
import numpy as np
import pandas as pd
import scipy as sp

from GELMMnet.utility.estimator import _nonparametric_cov_bootstrap, _sandwich_estimator

np.seterr(all="ignore")


from sklearn.preprocessing import scale

from GELMMnet.utility.inference import max_l1, _eval_neg_log_likelihood, _optimize_gelnet, _predict, _parameter_search, \
    _rmse, _corr, _cv_grid_search, alpha_grid
from GELMMnet.utility.postselection import _tg_limits, _tg_pval, _tg_interval


class GELMMnet(object):
    """
    Generalized network-based elastic-net linear mixed model

    \min_\beta~\frac{1}{2}\sum_{i=1}^N (y_i - X_i\beta)^2 + \alpha*\lambda\|\beta\|_1 + \frac{(1-\lambda)\alpha}{2}\beta^TP\beta

    1) We first infer sigma_g and sigma_e based on a Null-model following Kang et al 2010.
    2) We than rotate y and S based on the eigendecomposition of K following Rakitsch et al 2013
       and Schelldorfer et al 2011.
    3) We than fit the weights with coordinate descent as the transformation renders the problem a "simple"
       elastic-net inference problem

    """

    def __init__(self, y, X, K=None, intercept=True):

        # first check for correct input
        assert X.shape[0] == y.shape[0], ValueError('dimensions do not match X({}), y({})'.format(X.shape[0]),y.shape[0])

        v = np.isnan(y)
        keep = True ^ v
        if v.sum():
            print("Cleaning the phenotype vector by removing %d individuals...\n" % (v.sum()))
            y = y[keep]
            X = X[keep, :]
            if K is not None:
                K = K[keep, :][:, keep]
                assert K.shape[0] == K.shape[1], ValueError('dimensions do not match')
                assert K.shape[0] == X.shape[0], ValueError('dimensions do not match')

        self.__keep = keep
        self.__n, self.__m = X.shape

        if y.ndim == 1:
            y = sp.reshape(y, (self.__n, 1))

        self.__islmm = K is not None
        self.__isIntercept = intercept

        self.y = y
        self.X = X
        self.K = K
        self.P = np.identity(self.__m)

        self.SUX = None
        self.SUy = None

        self.w = np.zeros(self.__m)
        self.b = np.mean(y) if self.__isIntercept else 0.0
        self.sigma = np.mean(y)
        self.ldelta = 0

        self.l1 = 1.0
        self.l2 = 1.0

        self.hyperparameter_grid = None
        self.min_idx = None
        self.cv_error_curve = None
        self.nfold = 5

    @property
    def Xtilde(self):
        return self.SUX

    @property
    def ytilde(self):
        return self.SUy

    def fit_null_model(self, numintervals=100, ldeltamin=-20, ldeltamax=20, debug=False):
        """
        Optimizes sigma_g and simga_e based on a grid search using the approach
        proposed bei Khang et al. 2010

        """
        n = self.__n
        S, U = sp.linalg.eigh(self.K)

        if debug:
            print("Optimizing")
            print("U:", U)
            print("U.T", U.T)
            print("y", self.y)

        Uy = sp.dot(U.T, self.y)
        nllgrid = sp.ones(numintervals + 1) * np.inf
        ldeltagrid = sp.arange(numintervals + 1) / (numintervals * 1.0) * (ldeltamax - ldeltamin) + ldeltamin
        nllmin = sp.inf

        # initial grid search
        for i in np.arange(numintervals + 1):
            nllgrid[i] = _eval_neg_log_likelihood(ldeltagrid[i], Uy, S)
            if debug:
                print("Init grid:", nllgrid[i], ldeltagrid[i])

        # find minimum
        nll_min = nllgrid.min()
        ldeltaopt_glob = ldeltagrid[nllgrid.argmin()]

        # more accurate search around the minimum of the grid search
        for i in sp.arange(numintervals - 1) + 1:
            if nllgrid[i] < nllgrid[i - 1] and nllgrid[i] < nllgrid[i + 1]:
                ldeltaopt, nllopt, iter, funcalls = sp.optimize.brent(_eval_neg_log_likelihood,
                                                                      (Uy, S),
                                                                      (ldeltagrid[i - 1],
                                                                       ldeltagrid[i],
                                                                       ldeltagrid[i + 1]),
                                                                      full_output=True)
                if nllopt < nllmin:
                    if debug:
                        print("Second grid:", nllopt, ldeltaopt)
                    nllmin = nllopt
                    ldeltaopt_glob = ldeltaopt

        # train lasso on residuals
        self.ldelta = ldeltaopt_glob
        delta0 = sp.exp(ldeltaopt_glob)
        Sdi = 1. / (S + delta0)
        Sdi_sqrt = sp.sqrt(Sdi)
        SUX = sp.dot(U.T, self.X)
        self.SUX = SUX * sp.tile(Sdi_sqrt, (self.__m, 1)).T
        SUy = sp.dot(U.T, self.y)
        self.SUy = SUy * sp.reshape(Sdi_sqrt, (n, 1))

        if debug:
            print("delta0", delta0)
            print("Sdi_sqrt", Sdi_sqrt)
            print("SUX", SUX)
            print("SUy", SUy)

    def fit(self, P, l1=None, l2=None, eps=1e-8, max_iter=1000):
        """
            Train complete model
        """

        if l1 is not None:
            self.l1 = l1
        if l2 is not None:
            self.l2 = l2

        self.P = P

        if self.SUX is None and self.__islmm:
            self.fit_null_model()

        X = self.SUX if self.__islmm else self.X
        y = self.SUy if self.__islmm else self.y
        n, m = X.shape
        w = self.w
        b = self.b
        l1 = self.l1
        l2 = self.l2

        assert P.shape[0] == m, ValueError("P's dimensions do not match")

        # initial parameter estimates
        S = sp.dot(X, w) + b
        Pw = sp.dot(P, w)

        w, b = _optimize_gelnet(y, X, P, l1, l2, S, Pw, n, m, max_iter, eps, w, b, self.__isIntercept)

        self.w = w
        self.b = b
        return b, w

    def kfoldFit(self, P, nfold=5, l1_ratio=(.1, .5, .7, .9, .95, .99, 1), alpha_nof=100, eps=1e-8, max_iter=10000, scale=0, debug=False):
        """
        optimizes l1 and l2 based on k-fold cv with grid search minimizing the MSE

        """
        self.P = P
        self.nfold = nfold
        if debug and self.__islmm:
            print("Correcting for population structure")

        if self.SUX is None and self.__islmm:
            self.fit_null_model()

        Xt = self.SUX if self.__islmm else self.X
        Xv = self.X
        y = self.SUy if self.__islmm else self.y
        yv = self.y
        w = self.w
        b = self.b
        delta = np.exp(self.ldelta)
        n, m = Xt.shape

        # needed for post-selection
        Xy = np.dot(Xt.T, y[:, 0])
        self.hyperparameter_grid = [(r*a, (1.-r)*a) for r in l1_ratio
                                    for a in alpha_grid(r, Xy, Xt.shape[0], alpha_nof)]

        min_idx, l1, l2, error, error_curve = _cv_grid_search(Xt, Xv, y, yv, P, delta, self.__isIntercept,
                                                     scale, self.hyperparameter_grid, nfold, max_iter, eps,
                                                              self.__islmm)

        if debug:
            print("Best Parameters:", l1, l2, "With error:", np.mean(error))

        # fitting on whole date with best parameter pair
        S = np.dot(Xt, w) + b
        Pw = np.dot(P, w)
        w, b = _optimize_gelnet(y, Xt, P, l1, l2, S, Pw, n, m, max_iter, eps, w, b, self.__isIntercept)
        yhat = _predict(Xv, yv, Xv, w, b, delta, self.__islmm)

        self.w = w
        self.b = b
        self.l1 = l1
        self.l2 = l2
        self.min_idx = min_idx
        self.cv_error_curve = error_curve

        if debug:
            print(yv[:,0][:10])
            print(yhat[:10])
            print("Training Correlation:", _corr(yv[:, 0], yhat)," Training Error:", _rmse(yv[:, 0], yhat))

        df = np.sum(w != 0) - 1
        sigma = np.sqrt(np.sum(np.power(yv[:, 0] - yhat, 2)))/(len(y) - df)
        self.sigma = sigma
        if debug:
            print("nominator:", np.sqrt(np.sum(np.power(yv[:, 0] - yhat, 2))))
            print("denominator:", (len(y) - df))
            print("sigma:", sigma)

        return l1, l2, sigma, error_curve

    def predict(self, X_tilde):
        """
        predicts the phenotype based on the trained model

        following Rasmussen and Williams 2006

        :param X_tilde: test matrix n_test x m
        :return: y_tilde the predicted phenotype
        """

        assert X_tilde.shape[1] == self.__m, ValueError("Dimension of input data does not match")

        w = self.w
        b = self.b
        y = self.y
        X = self.X
        delta = np.exp(self.ldelta)

        return _predict(X, y, X_tilde, w, b, delta, self.__islmm)

    def post_selection_analysis(self, alpha=0.1, compute_intervals=False, gridrange=(-100, 100), tol_beta=1e-5,
                                tol_kkt=0.1, sigma=None):
        """
        implements the post selection analysis proposed by

        Lee, J. D., Sun, D. L., Sun, Y., & Taylor, J. E. (2016).
        Exact post-selection inference, with application to the lasso.
        The Annals of Statistics, 44(3), 907-927.

        only applicable for fixed hyperparameters! Hyperparameter optimization will effect the correctness of the
        p-values

        See: https://github.com/selective-inference/Python-software

        :param float alpha: The (1-alpha) selective confidence intervals
        :return: DataFrame with one entry per active variable. Columns are

        'variable', 'pval', 'lasso', 'beta','Zscore', 'lower_ci', 'upper_ci', 'lower_trunc', 'upper_trunc', 'sd'.

        """
        if self.hyperparameter_grid is not None:
            Warning("p-values might be strongly effected by hyperparameter estimation. \
                    Consider using 'kfold_post_selection_analysis' instead.")

        if self.SUX is None and self.__islmm:
            RuntimeError("The model has not been trained yet")

        X = self.SUX if self.__islmm else self.X
        n,m = X.shape
        y = self.SUy if self.__islmm else self.y[:,0]
        w = self.w
        P = self.P
        l2 = self.l2
        l1 = self.l1
        _sigma = np.std(y)

        if sigma is not None:
            _sigma = sigma
        if self.sigma is not None:
            _sigma = self.sigma

        if self.__isIntercept:
            y = scale(y, with_std=False)
            X = scale(X, with_std=False)

        # compute the hessian of the active set M
        # H(M) = (XT_M*X_M + l2*P)^-1
        H = (np.dot(np.transpose(X), X) + l2*P)

        # Check KKT condition
        g = (np.dot(H, w) - np.dot(np.transpose(X), y))/l1
        if np.any(np.fabs(g) > 1.+tol_kkt * np.sqrt(np.sum(np.power(y, 2)))):
            Warning("Beta does not satisfy the KKT conditions (to within specified tolerances)")

        active = np.where(np.fabs(w) > tol_beta / np.sqrt(np.sum(np.power(X, 2), axis=0)))[0]

        active_signs = np.sign(w[active])

        if not active.size:
            Warning("Model is empty")
            return None

        if np.any(np.sign(g[active]) != active_signs):
            Warning(
                "Solution beta does not satisfy the KKT conditions (to within specified tolerances). " +
                "You might try rerunning GELMMnet with a lower setting of the 'eps' parameter, " +
                "for a more accurate convergence."
            )

        XM = X[:, active]
        H_AA = H[active][:, active]
        H_AAinv = np.linalg.pinv(H_AA)

        D = np.diag(active_signs)

        '''
        The active set is defined as:

        A1(M,s) = -diag(s)(X^T_M*X_M + l2*P)^-1*X^T_M
        b1(M,s) = -l1*diag(s)(X^T_M*X_M + l2*P)^-1*s
        '''
        A = np.dot(D, np.dot(H_AAinv, np.transpose(XM)))

        b = l1 * np.dot(D, np.dot(H_AAinv, active_signs))
        tol_poly = 0.01
        if np.min(np.dot(A, y) - b) < -tol_poly * np.sqrt(np.sum(np.power(y, 2))):
            ValueError("Polyhedral constraints not satisfied; you must recompute beta more accurately.")


        ################################################################################################################
        # P-value and CI calculations

        k = len(active)
        M = np.dot(H_AAinv, np.transpose(XM))

        result = []
        print("Starting p-value calculation")
        for j in range(k):

            vj = M[j]
            mj = np.sqrt(np.sum(np.power(vj, 2)))
            vj = vj / mj
            sign = np.sign(np.sum(vj*y))
            vj = sign * vj

            # calculate p-value
            vlo, vup, sd, estimate = _tg_limits(y, A, b, vj, np.diag(np.ones(n)*np.power(_sigma, 2)))
            _pval = _tg_pval(estimate, vlo, vup, sd)

            # two-sided; I think calculated p-value is one-sided
            #_pval = 2. * min(_pval, 1. - _pval)

            vmat = vj * mj * sign

            if compute_intervals:
                _interval, tailarea = _tg_interval(estimate, vlo, vup, sd, alpha,
                                                   gridrange=gridrange, flip=sign == -1)

                ci = [x*mj for x in _interval]
            else:
                ci = [np.nan, np.nan]
                tailarea = [np.nan, np.nan]

            sd = _sigma * np.sqrt(np.sum(np.power(vmat, 2)))
            coef0 = np.dot(vmat, y)
            result.append((active[j],
                           _pval,
                           coef0,
                           w[active[j]],
                           coef0 / sd,
                           ci[0],
                           ci[1],
                           tailarea[0],
                           tailarea[1],
                           sd))

        df = pd.DataFrame(index=active,
                      data=OrderedDict([(n, d) for n, d in zip(['variable',
                                                         'pval',
                                                         'coef',
                                                         'beta',
                                                         'Zscore',
                                                         'lower_confidence',
                                                         'upper_confidence',
                                                         'lower_trunc',
                                                         'upper_trunc',
                                                         'sd'],
                                                        np.array(result).T)])).set_index('variable')
        self.summary = df
        return df

    def kfold_post_selection_analysis(self, alpha=0.1, compute_intervals=False, gridrange=(-100, 100),
                                      tol_beta=1e-5, tol_kkt=0.1, nsamples=500):
        """
        Implements the grid search and selection process correction procedure proposed by


        Markovic, J., Xia, L., & Taylor, J. (2017).
        Adaptive p-values after cross-validation.
        arXiv pre-print arXiv:1703.06559.

        :param alpha:
        :param compute_intervals:
        :param gridrange:
        :param tol_beta:
        :param tol_kkt:
        :return:
        """
        raise NotImplementedError()

        Xt = self.SUX if self.__islmm else self.X
        Xv = self.X
        yt = self.SUy if self.__islmm else self.y
        yv = self.y
        w = self.w
        b = self.b
        delta = np.exp(self.ldelta)
        P = self.P
        l2 = self.l2
        l1 = self.l1
        n, m = Xt.shape

        active = np.where(np.fabs(w) > tol_beta / np.sqrt(np.sum(np.power(Xt, 2), axis=0)))[0]
        inactive = np.where(np.fabs(w) <= tol_beta / np.sqrt(np.sum(np.power(Xt, 2), axis=0)))[0]
        active_signs = np.sign(w[active])

        nactive = len(active)

        if not nactive:
            raise ValueError("Solution is the null model. No variables were selected during inference")

        # setup model selection constraints
        H = (np.dot(np.transpose(Xt), Xt) + l2 * P)
        H_AA = H[active][:, active]
        H_AAinv = np.linalg.pinv(H_AA)
        XM = Xt[:, active]
        D = np.diag(active_signs)

        # we probably also need the inactive constraints?!
        model_A = np.dot(D, np.dot(H_AAinv, np.transpose(XM)))
        model_a = l1 * np.dot(D, np.dot(H_AAinv, active_signs))

        # TODO: Dont exactly know what the one_step estimator is......
        one_step = None

        # compute covariance of selected parameters with CV error curve
        cov = _nonparametric_cov_bootstrap(Xt, Xv, yt, yv, l2, P, delta,
                                           self.hyperparameter_grid, self.__isIntercept, active, nsamples)

        Sigma = _sandwich_estimator(Xt, yt, l2, P, w, active, inactive, 2000)
        A = np.dot(cov[1].T, np.linalg.pinv(Sigma))
        residual = np.array(list(self.cv_error_curve.values())) - np.dot(A, one_step)


        # setup CV constraints
        cv_error_len = len(self.cv_error_curve)
        lam_keep_randomized = np.zeros(cv_error_len, np.bool)
        lam_keep_randomized[self.min_idx] = 1
        B = -np.identity(cv_error_len)
        B += (np.multiply.outer(lam_keep_randomized, np.ones_like(lam_keep_randomized))).T

        keep = np.ones(cv_error_len, np.bool)
        keep[self.min_idx] = 0
        B = B[keep]
        C = B.dot(A)

        # TODO: put all constraints together LHS and RHS
        cv_a = -np.dot(B, residual)



        ################################################################################################################
        # P-value and CI calculations

        k = len(active)
        M = np.dot(H_AAinv, np.transpose(XM))

        result = []
        for j in range(k):

            vj = M[j]
            mj = np.sqrt(np.sum(np.power(vj, 2)))
            vj = vj / mj
            sign = np.sign(np.sum(vj*yt))
            vj = sign * vj

            # calculate p-value
            vlo, vup, sd, estimate = _tg_limits(yt, A, b, vj, Sigma)
            _pval = _tg_pval(estimate, vlo, vup, sd)

            # two-sided; I think calculated p-value is one-sided
            _pval = 2 * min(_pval, 1 - _pval)

            vmat = vj * mj * sign

            if compute_intervals:
                _interval, tailarea = _tg_interval(estimate, vlo, vup, sd, alpha,
                                                   gridrange=gridrange, flip=sign == -1)

                ci = [x*mj for x in _interval]
            else:
                ci = [np.nan, np.nan]

            # TODO: this has to be vecotrized as Sigma is now a covariance matrix
            sd = _sigma * np.sqrt(np.sum(np.power(vmat, 2)))
            coef0 = np.dot(vmat, yt)
            result.append((active[j],
                           _pval,
                           coef0,
                           w[active[j]],
                           coef0 / sd,
                           ci[0],
                           ci[1],
                           tailarea[0],
                           tailarea[1],
                           sd))

        df = pd.DataFrame(index=active,
                          data=OrderedDict([(n, d) for n, d in zip(['variable',
                                                             'pval',
                                                             'coef',
                                                             'beta',
                                                             'Zscore',
                                                             'lower_confidence',
                                                             'upper_confidence',
                                                             'lower_trunc',
                                                             'upper_trunc',
                                                             'sd'],
                                                            np.array(result).T)])).set_index('variable')
        self.summary = df
        return df











