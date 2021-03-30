# -*- coding: utf-8 -*-
# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License version 3 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                        documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Hessian matrix approximations from gradient pairs
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from builtins import range, str, zip

from future import standard_library
from numpy import array, atleast_2d, concatenate, cumsum
from numpy import diag as np_diag
from numpy import dot, eye, inf, sqrt, trace, zeros
from numpy.linalg import LinAlgError, cholesky, inv, multi_dot, norm
from numpy.matlib import repmat
from scipy.optimize import leastsq

standard_library.install_aliases()

from gemseo import LOGGER


class HessianApproximation(object):
    """Abstract class for hessian approximations from optimization history"""

    def __init__(self, history):
        """
        Constructor
        :param history: the optimization history
        """
        self.history = history
        self.x_ref = None
        self.fgrad_ref = None
        self.f_ref = None
        self.b_mat_history = []
        self.h_mat_history = []

    def get_x_grad_history(
        self,
        funcname,
        first_iter=0,
        last_iter=0,
        at_most_niter=-1,
        func_index=None,
        normalize_design_space=False,
        design_space=None,
    ):
        """Get gradient history and design variables history of gradient
        evaluations

        :param funcname: function name
        :param first_iter: first iteration after which the history is
            extracted (Default value = 0)
        :param last_iter: last iteration before which the history is
            extracted (Default value = 0)
        :param at_most_niter: maximum number of iterations to take
            (Default value = -1)
        :param func_index: Default value = None)
        :param normalize_design_space: if True, scale the values
            to work in a normalized design space (x between 0 and 1)
        :param design_space: the design space used to scale all values
            mandatory if normalize_design_space==True

        """
        x_grad_hist, x_hist = self.history.get_func_grad_history(funcname, x_hist=True)
        if normalize_design_space:
            (
                x_hist,
                x_grad_hist,
            ) = self._normalize_x_g(x_hist, x_grad_hist, design_space)
        assert len(x_grad_hist) == len(x_hist)
        n_pairs = len(x_grad_hist)
        if n_pairs < 2:
            raise ValueError(
                "Cannot build approximation for function : {}"
                " because its gradient history "
                "is too small : {}.".format(funcname, n_pairs)
            )
        #         LOGGER.info("Building Hessian approximation with " +
        #                     str(n_pairs) + " pairs of x and gradients")
        x_grad_hist = array(x_grad_hist)
        x_hist = array(x_hist)
        n_iter, nparam = x_hist.shape
        grad_shape = x_grad_hist.shape
        n_iter_g = grad_shape[0]
        nparam_g = grad_shape[-1]
        # Function is a vector, jacobian is a 2D matrix
        if len(grad_shape) == 3:
            if func_index is None:
                raise ValueError(
                    "Function {} has a vector output "
                    "then function index of output "
                    "must be specified.".format(funcname)
                )
            if func_index < 0 or func_index >= grad_shape[1]:
                raise ValueError(
                    "Function {} has a vector output of size {},"
                    "function index {} is out of range"
                    ".".format(funcname, grad_shape[1], func_index)
                )
            x_grad_hist = x_grad_hist[:, func_index, :]

        if (n_iter != n_iter_g) or (nparam_g != nparam):
            raise ValueError(
                "Inconsistent gradient and design" " variables optimization history"
            )
        if last_iter == 0:
            x_hist = x_hist[first_iter:, :]
            x_grad_hist = x_grad_hist[first_iter:, :]
        else:
            x_hist = x_hist[first_iter:last_iter, :]
            x_grad_hist = x_grad_hist[first_iter:last_iter, :]

        n_iter = x_hist.shape[0]
        if 0 < at_most_niter < n_iter:
            x_hist = x_hist[n_iter - at_most_niter :, :]
            x_grad_hist = x_grad_hist[n_iter - at_most_niter :, :]
        n_iter, nparam = x_hist.shape
        if n_iter < 2 or nparam == 0:
            raise ValueError(
                "Insufficient optimization history size, "
                "niter={} nparam = {}.".format(n_iter, nparam)
            )
        self.x_ref = x_hist[-1]
        self.fgrad_ref = x_grad_hist[-1]
        if last_iter == 0:
            self.f_ref = array(self.history.get_func_history(funcname))[-1]
        else:
            self.f_ref = array(self.history.get_func_history(funcname))[:last_iter][-1]
        return x_hist, x_grad_hist, n_iter, nparam

    @staticmethod
    def _normalize_x_g(x_hist, x_grad_hist, design_space):
        """
         scale the values
            to work in a normalized design space (x between 0 and 1)
        :param design_space: the design space used to scale all values
        """
        if design_space is None:
            raise ValueError(
                "Design space must be provided when using "
                "a normalize_design_space option!"
            )
        unnormalize_vect = design_space.unnormalize_vect

        def normalize_gradient(x_vect):
            """Unnormalize a gradient"""
            return unnormalize_vect(x_vect, minus_lb=False, no_check=True)

        x_scaled, grad_scaled = [], []
        for x_v, g_v in zip(x_hist, x_grad_hist):
            x_scaled.append(design_space.normalize_vect(x_v))
            grad_scaled.append(normalize_gradient(g_v))
        return x_scaled, grad_scaled

    @staticmethod
    def get_s_k_y_k(x_hist, x_grad_hist, iteration):
        """Generate the s_k and y_k terms, respectively design variable
        difference and gradients difference between iterates

        :param x_hist: design variables history array
        :param x_grad_hist: gradients history array
        :param iteration: iteration number for which the pair must be generated

        """
        n_iter = x_grad_hist.shape[0]
        if iteration >= n_iter:
            raise ValueError(
                "Iteration {} is higher than number of gradients "
                "in the database : {}.".format(iteration, n_iter)
            )
        s_k = atleast_2d(x_hist[iteration + 1] - x_hist[iteration]).T
        y_k = atleast_2d(x_grad_hist[iteration + 1] - x_grad_hist[iteration]).T
        return s_k, y_k

    @staticmethod
    def iterate_s_k_y_k(x_hist, x_grad_hist):
        """Generate the s_k and y_k terms, respectively design variable
        difference and gradients difference between iterates

        :param x_hist: design variables history array
        :param x_grad_hist: gradients history array

        """
        n_iter = x_hist.shape[0]
        for k in range(n_iter - 1):
            s_k, y_k = HessianApproximation.get_s_k_y_k(x_hist, x_grad_hist, k)
            yield s_k, y_k

    def build_approximation(
        self,
        funcname,
        save_diag=False,
        first_iter=0,
        last_iter=-1,
        b_mat0=None,
        at_most_niter=-1,
        return_x_grad=False,
        func_index=None,
        save_matrix=False,
        scaling=False,
        normalize_design_space=False,
        design_space=None,
    ):  # pylint: disable=W0221
        """Builds the hessian approximation B.

        :param funcname: function name
        :param save_diag: if True, returns the list of diagonal approximations
            (Default value = False)
        :param first_iter: first iteration after which the history is extracted
            (Default value = 0)
        :param last_iter: last iteration before which the history is extracted
            (Default value = -1)
        :param b_mat0: initial approximation matrix (Default value = None)
        :param at_most_niter: maximum number of iterations to take
            (Default value = -1)
        :param return_x_grad: if True, also returns the last gradient and x
            (Default value = False)
        :param func_index: Default value = None)
            (Default value = False)
        :param normalize_design_space: if True, scale the values
            to work in a normalized design space (x between 0 and 1)
        :param design_space: the design space used to scale all values
            mandatory if normalize_design_space==True
        :returns: the B matrix, its diagonal,
            and eventually the x and grad history pairs
            used to build B, if return_x_grad=True,
            otherwise, None and None are returned for args consistency

        """
        normalize_ds = normalize_design_space
        x_hist, x_grad_hist, _, _ = self.get_x_grad_history(
            funcname,
            first_iter,
            last_iter,
            at_most_niter,
            func_index,
            normalize_ds,
            design_space,
        )
        if b_mat0 is None:
            for s_k, y_k in self.iterate_s_k_y_k(x_hist, x_grad_hist):
                break
            last_n_grad = x_grad_hist.shape[0] - 2
            s_k, y_k = self.get_s_k_y_k(x_hist, x_grad_hist, last_n_grad)
            alpha = dot(y_k.T, s_k) / dot(y_k.T, y_k)
            b_mat = (1.0 / alpha) * eye(len(x_grad_hist[0]))
        elif b_mat0.size == 0:
            n_x = len(x_hist[0])
            b_mat = zeros((n_x, n_x))
        else:
            b_mat = b_mat0
        diag = []
        for s_k, y_k in self.iterate_s_k_y_k(x_hist, x_grad_hist):
            self.iterate_approximation(b_mat, s_k, y_k, scaling=scaling)
            if save_diag:
                diag.append(np_diag(b_mat).copy())
            if save_matrix:
                self.b_mat_history.append(b_mat.copy())
        if return_x_grad:
            return b_mat, diag, x_hist[-1, :], x_grad_hist[-1, :]
        return b_mat, diag, None, None

    @staticmethod
    def compute_scaling(hessk, hessk_dsk, dskt_hessk_dsk, dyk, dyt_dsk):
        """Compute scaling
        :param hessk: previous approximation
        :param hessk_s: product between hessk and dsk
        :param dskt_hessk_sk: product between dsk^t, hessk and dsk
        :param dyk: gradients difference between iterates
        :param dyt_dsk: product between dyk^t and dsk^t
        """
        coeff1 = (len(hessk_dsk) - 1) / (
            trace(hessk) - norm(hessk_dsk) ** 2 / dskt_hessk_dsk
        )
        coeff2 = dyt_dsk / norm(dyk) ** 2
        return coeff1, coeff2

    @staticmethod
    def iterate_approximation(hessk, dsk, dyk, scaling=False):
        """BFGS iteration from step k to step k+1

        :param hessk: previous approximation
        :param dsk: design variable difference between iterates
        :param dyk: gradients difference between iterates
        :param scaling: do scaling step
        :returns: updated approximation

        """
        dyt_dsk = dot(dyk.T, dsk)
        hessk_dsk = dot(hessk, dsk)
        dskt_hessk_dsk = multi_dot((dsk.T, hessk, dsk))
        # Build the next approximation:
        b_first_term = hessk - multi_dot((hessk, dsk, dsk.T, hessk)) / dskt_hessk_dsk
        b_second_term = dot(dyk, dyk.T) / dyt_dsk
        if not scaling:
            hessk[:, :] = b_first_term + b_second_term
        else:
            c_1, c_2 = HessianApproximation.compute_scaling(
                hessk, hessk_dsk, dskt_hessk_dsk, dyk, dyt_dsk
            )
            hessk[:, :] = c_1 * b_first_term + c_2 * b_second_term

    def build_inverse_approximation(
        self,
        funcname,
        save_diag=False,
        first_iter=0,
        last_iter=-1,
        h_mat0=None,
        at_most_niter=-1,
        return_x_grad=False,
        func_index=None,
        save_matrix=False,
        factorize=False,
        scaling=False,
        angle_tol=1e-5,
        step_tol=1e10,
        normalize_design_space=False,
        design_space=None,
    ):
        """Builds the inversed hessian approximation H

        :param funcname: function name
        :param save_diag: if True, returns the list of diagonal approximations
            (Default value = False)
        :param first_iter: first iteration after which the history is extracted
            (Default value = 0)
        :param last_iter: last iteration before which the history is extracted
            (Default value = -1)
        :param h_mat0: initial inverse approximation matrix
            (Default value = None)
        :param at_most_niter: maximum number of iterations to take
            (Default value = -1)
        :param return_x_grad: if True, also returns the last gradient and x
            (Default value = False)
        :param func_index: Default value = None)
        :param normalize_design_space: if True, scale the values
            to work in a normalized design space (x between 0 and 1)
        :param design_space: the design space used to scale all values
            mandatory if normalize_design_space==True
        :returns: the H matrix, its diagonal,
            and eventually the x and grad history pairs
            used to build H, if return_x_grad=True,
            otherwise, None and None are returned for args consistency

        """
        normalize_ds = normalize_design_space
        x_hist, x_grad_hist, _, _ = self.get_x_grad_history(
            funcname,
            first_iter,
            last_iter,
            at_most_niter,
            func_index,
            normalize_ds,
            design_space,
        )
        b_mat = None  # to become the Hessian approximation B, optionally
        h_factor = None  # to become a matrix G such that H = G*G', optionally
        b_factor = None  # to become the inverse of the matrix G
        if h_mat0 is None:
            for s_k, y_k in self.iterate_s_k_y_k(x_hist, x_grad_hist):
                break
            last_n_grad = x_grad_hist.shape[0] - 2
            s_k, y_k = self.get_s_k_y_k(x_hist, x_grad_hist, last_n_grad)
            alpha = dot(y_k.T, s_k) / dot(y_k.T, y_k)
            n_x = len(x_grad_hist[0])
            h_mat = alpha * eye(n_x)
            b_mat = 1.0 / alpha * eye(n_x)
            if factorize:
                h_factor = sqrt(alpha) * eye(n_x)
                b_factor = eye(n_x) / sqrt(alpha)
        elif len(h_mat0) == 0:
            n_x = len(x_hist[0])
            h_mat = zeros((n_x, n_x))
            b_mat = zeros((n_x, n_x))
            if factorize:
                h_factor = zeros((n_x, n_x))
                b_factor = zeros((n_x, n_x))
        else:
            h_mat = h_mat0
            try:
                b_mat = inv(h_mat)
            except LinAlgError:
                raise LinAlgError("The inversion of h_mat failed")
            if factorize or scaling:
                try:
                    h_factor = cholesky(h_mat)
                    b_factor = cholesky(b_mat).T
                except LinAlgError:
                    raise LinAlgError(
                        "The Cholesky decomposition of h_factor" " or b_factor failed"
                    )
        diag = []
        count = 0
        k = 0
        for s_k, y_k in self.iterate_s_k_y_k(x_hist, x_grad_hist):
            k = k + 1
            if dot(s_k.T, y_k) > angle_tol and norm(y_k, inf) < step_tol:
                count = count + 1
                self.iterate_inverse_approximation(
                    h_mat,
                    s_k,
                    y_k,
                    h_factor,
                    b_mat,
                    b_factor,
                    factorize=factorize,
                    scaling=scaling,
                )
            if save_diag:
                diag.append(np_diag(h_mat).copy())
            if save_matrix:
                self.h_mat_history.append(h_mat.copy())
        if return_x_grad:
            return h_mat, diag, x_hist[-1, :], x_grad_hist[-1, :], None, None, None
        return h_mat, diag, None, None, h_factor, b_mat, b_factor

    @staticmethod
    def compute_corrections(x_hist, x_grad_hist):
        """ Computes the corrections from the history. """
        n_iter = x_hist.shape[0]
        x_corr = x_hist[1:n_iter].T - x_hist[: n_iter - 1].T
        grad_corr = x_grad_hist[1:n_iter].T - x_grad_hist[: n_iter - 1].T
        return x_corr, grad_corr

    @staticmethod
    def rebuild_history(x_corr, x_0, grad_corr, g_0):
        """ Computes the history from the corrections. """
        # Rebuild the argument history:
        x_hist = repmat(x_0, x_corr.shape[1], 1) + cumsum(x_corr.T, axis=0)
        x_hist = concatenate((atleast_2d(x_0), x_hist), axis=0)
        # Rebuild the gradient history
        x_grad_hist = repmat(g_0, grad_corr.shape[1], 1) + cumsum(grad_corr.T, axis=0)
        x_grad_hist = concatenate((atleast_2d(g_0), x_grad_hist), axis=0)
        return x_hist, x_grad_hist

    @staticmethod
    def iterate_inverse_approximation(
        h_mat,
        s_k,
        y_k,
        h_factor=None,
        b_mat=None,
        b_factor=None,
        factorize=False,
        scaling=False,
    ):
        """Inverse BFGS iteration

        :param h_mat: previous approximation
        :param s_k: design variable difference between iterates
        :param y_k: gradients difference between iterates
        :returns: updated inverse approximation

        """
        # Compute the two terms of the non-scaled updated matrix:
        yts = dot(y_k.T, s_k)
        proj = eye(len(s_k)) - dot(s_k, y_k.T) / yts
        h_first_term = multi_dot((proj, h_mat, proj.T))
        h_second_term = dot(s_k, s_k.T) / yts
        b_s = dot(b_mat, s_k)
        st_b_s = dot(s_k.T, b_s)
        # Compute the scaling coefficients:
        if scaling:
            coeff1, coeff2 = HessianApproximation.compute_scaling(
                b_mat, b_s, st_b_s, y_k, yts
            )
        else:
            coeff1, coeff2 = 1.0, 1.0
        # Update the inverse approximation H and, optionally, the factor G:
        h_mat[:, :] = h_first_term / coeff1 + h_second_term / coeff2
        if factorize:
            sst_b = dot(s_k, b_s.T)
            left = proj / sqrt(coeff1) + sst_b / sqrt(coeff2 * yts * st_b_s)
            h_factor[:, :] = dot(left, h_factor)
            #             b_factor[:, :] = dot(eye(len(s_k)) - sstB.T / stBs / sqrt(coeff1)
            #                                  + dot(y_k, s_k.T)
            #                                  / sqrt(coeff2 * stBs * yts),
            #                                  b_factor)
            right = sqrt(coeff1) * (eye(len(s_k)) - sst_b / st_b_s)
            right += sqrt(coeff2) * dot(s_k, y_k.T) / sqrt(st_b_s * yts)
            b_factor[:, :] = dot(b_factor, right)

        # Update the Hessian approximation:
        b_first_term = b_mat - multi_dot((b_s, b_s.T)) / st_b_s
        b_second_term = dot(y_k, y_k.T) / yts
        b_mat[:, :] = coeff1 * b_first_term + coeff2 * b_second_term


#             b_mat[:, :] = multi_dot((proj.T, b_mat, proj)) \
#                 + dot(y_k, y_k.T) / yts


class BFGSApprox(HessianApproximation):

    """Builds a BFGS approximation from optimization history"""

    @staticmethod
    def iterate_s_k_y_k(x_hist, x_grad_hist):
        """Generate the s_k and y_k terms, respectively design variable
        difference and gradients difference between iterates

        :param x_hist: design variables history array
        :param x_grad_hist: gradients history array

        """
        n_iter = x_hist.shape[0]
        for k in range(n_iter - 1):
            s_k, y_k = BFGSApprox.get_s_k_y_k(x_hist, x_grad_hist, k)
            # All pairs curvatures shall be positive
            # if dot(s_k.T, y_k) > 0.:
            if dot(s_k.T, y_k) > 1e-16 * dot(y_k.T, y_k):
                yield s_k, y_k


class SR1Approx(HessianApproximation):

    """Builds a Symmetric Rank One approximation from optimization history"""

    EPSILON = 1e-8

    @staticmethod
    def iterate_approximation(b_mat, s_k, y_k, scaling=False):
        """SR1 iteration

        :param b_mat: previous approximation
        :param s_k: design variable difference between iterates
        :param y_k: gradients difference between iterates
        :param scaling: do scaling sep
        :returns: updated approximation

        """
        d_mat = y_k - multi_dot((b_mat, s_k))
        den = multi_dot((d_mat.T, s_k))
        if abs(den) > SR1Approx.EPSILON * norm(s_k) * norm(d_mat):
            b_mat[:, :] = b_mat + multi_dot((d_mat, d_mat.T)) / den
        else:
            LOGGER.debug(
                "Denominator of SR1 update is too small, " "update s_kipped %s.", den
            )


class LSTSQApprox(HessianApproximation):

    """Builds a Least squares approximation from optimization history"""

    def build_approximation(
        self,
        funcname,
        save_diag=False,
        first_iter=0,
        last_iter=-1,
        b_mat0=None,
        at_most_niter=-1,
        return_x_grad=False,
        scaling=False,
        func_index=-1,
        normalize_design_space=False,
        design_space=None,
    ):
        """Builds the hessian approximation

        :param funcname: function name
        :param save_diag: if True, returns the list of diagonal approximations
            (Default value = False)
        :param first_iter: first iteration after which the history is extracted
            (Default value = 0)
        :param last_iter: last iteration before which the history is extracted
            (Default value = -1)
        :param b_mat0: initial approximation matrix (Default value = None)
        :param at_most_niter: Default value = -1)
        :param return_x_grad: Default value = False)
        :param func_index: Default value = -1)
        :param normalize_design_space: if True, scale the values
            to work in a normalized design space (x between 0 and 1)
        :param design_space: the design space used to scale all values
            mandatory if normalize_design_space==True
        :returns: the B matrix, its diagonal,
            and eventually the x and grad history pairs
            used to build B, if return_x_grad=True,
            otherwise, None and None are returned for args consistency
        """
        x_hist, x_grad_hist, n_iter, nparam = self.get_x_grad_history(
            funcname,
            first_iter,
            last_iter,
            at_most_niter,
            func_index=func_index,
            normalize_design_space=normalize_design_space,
            design_space=design_space,
        )
        sec_dim = max(nparam, n_iter)
        diag = []
        assert len(x_grad_hist) == len(x_hist)

        def y_to_b(y_vars):
            """Reshapes the approximation from vector to matrix

            :param y: the vector approximation
            :param y_vars: returns: the matrix shaped approximation
            :returns: the matrix shaped approximation

            """
            y_mat = y_vars.reshape((nparam, nparam))
            return y_mat + y_mat.T

        def func(y_vars):
            """Create the least square function

            :param y: the current approximation vector
            :param y_vars: returns: the estimated error vector
            :returns: the estimated error vector

            """
            b_mat_current = y_to_b(y_vars)
            err = zeros((nparam, sec_dim))
            for i in range(n_iter):
                x_l = x_hist[i] - self.x_ref
                err[:, i] = dot(b_mat_current, x_l) - x_grad_hist[i]
            err = err.reshape(nparam * sec_dim)
            if n_iter < nparam:  #
                err += y_vars
            return err

        y_0 = zeros(nparam * nparam)
        LOGGER.debug("Start least squares problem..")
        y_opt, ier = leastsq(func, x0=y_0)  # , cov_x, infodict, mesg, ier
        LOGGER.debug("End least squares, msg=%s", str(ier))
        b_mat = y_to_b(y_opt)
        if return_x_grad:
            return b_mat, diag, x_hist[-1, :], x_grad_hist[-1, :]
        return b_mat, diag, None, None
