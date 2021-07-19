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
"""Quantification of robustness of the optimum to variables perturbations."""
from __future__ import division, unicode_literals

import numpy as np
from numpy.random import multivariate_normal

from gemseo.post.core.hessians import BFGSApprox, LSTSQApprox, SR1Approx


class RobustnessQuantifier(object):
    """classdocs."""

    AVAILABLE_APPROXIMATIONS = ["BFGS", "SR1", "LEAST_SQUARES"]

    def __init__(self, history, approximation_method="SR1"):
        """Constructor.

        :param history: an approximation history.
        :param approximation_method: an approximation method for the Hessian.
        """
        self.history = history
        if approximation_method not in self.AVAILABLE_APPROXIMATIONS:
            raise ValueError(
                "Unknown hessian approximation method "
                + str(approximation_method)
                + " the available ones are:"
                + str(self.AVAILABLE_APPROXIMATIONS)
            )
        if approximation_method == "SR1":
            approx_class = SR1Approx
        elif approximation_method == "BFGS":
            approx_class = BFGSApprox
        elif approximation_method == "LEAST_SQUARES":
            approx_class = LSTSQApprox
        self.approximator = approx_class(history)
        self.b_mat = None
        self.x_ref = None
        self.f_ref = None
        self.fgrad_ref = None

    def compute_approximation(
        self,
        funcname,
        first_iter=0,
        last_iter=0,
        b0_mat=None,
        at_most_niter=-1,
        func_index=None,
    ):
        """Builds the BFGS approximation for the hessian.

        :param at_most_niter: maximum number of iterations to take
            (Default value = -1).
        :param funcname: param first_iter:  (Default value = 0).
        :param last_iter: Default value = 0).
        :param b0_mat: Default value = None).
        :param func_index: Default value = None).
        :param first_iter:  (Default value = 0).
        """

        self.b_mat, _, x_ref, grad_ref = self.approximator.build_approximation(
            funcname=funcname,
            first_iter=first_iter,
            last_iter=last_iter,
            b_mat0=b0_mat,
            at_most_niter=at_most_niter,
            return_x_grad=True,
            func_index=func_index,
        )
        self.x_ref = x_ref
        self.f_ref = self.approximator.f_ref
        self.fgrad_ref = grad_ref
        return self.b_mat

    def compute_expected_value(self, expect, cov):
        """computes 1/2*E(e.T B e) , where e is a vector of expected values expect and
        covariance matrix cov.

        :param expect: the expected value of inputs.
        :param cov: the covariance matrix of inputs.
        """
        n_vars = len(expect)
        if cov.shape != (n_vars, n_vars):
            raise ValueError("Inconsistent expect and covariance matrices shapes")
        if self.b_mat is None:
            raise ValueError(
                "Build Hessian approximation before"
                + " computing expected_value_offset"
            )
        b_approx = 0.5 * self.b_mat
        exp_val = np.trace(np.dot(b_approx, cov))
        exp_val += np.linalg.multi_dot(
            ((expect - self.x_ref).T, b_approx, (expect - self.x_ref))
        )
        return exp_val

    def compute_variance(self, expect, cov):
        """computes 1/2*E(e.T B e), where e is a vector of expected values expect and
        covariance matrix cov.

        :param expect: the expected value of inputs.
        :param cov: the covariance matrix of inputs.
        """
        if self.b_mat is None:
            raise ValueError(
                "Build Hessian approximation" + " before computing variance"
            )
        n_vars = len(expect)
        if cov.shape != (n_vars, n_vars):
            raise ValueError("Inconsistent expect and covariance matrices shapes")
        b_approx = 0.5 * self.b_mat
        mu_cent = expect - self.x_ref
        v_mat = np.linalg.multi_dot((b_approx, cov, b_approx, cov))
        v_mat += 4 * np.linalg.multi_dot((mu_cent.T, b_approx, cov, b_approx, mu_cent))
        var = 2 * np.trace(v_mat)
        return var

    def compute_function_approximation(self, x_vars):
        """Computes a second order approximation of the function.

        :param x_vars: the point on which the approximation is evaluated.
        :param x_vars: x vars.
        """
        if self.b_mat is None or self.x_ref is None:
            raise ValueError(
                "Build Hessian approximation before"
                + " computing function approximation"
            )
        x_l = x_vars - self.x_ref
        return (
            0.5 * np.linalg.multi_dot((x_l.T, self.b_mat, x_l))
            + np.dot(self.fgrad_ref.T, x_l)
            + self.f_ref
        )

    def compute_gradient_approximation(self, x_vars):
        """Computes a first order approximation of the gradient based on the hessian.

        :param x_vars: the point on which the approximation is evaluated.
        """
        if self.b_mat is None or self.fgrad_ref is None:
            raise ValueError(
                "Build Hessian approximation before"
                + " computing function approximation"
            )
        x_l = x_vars - self.x_ref
        return np.dot(self.b_mat, x_l) + self.fgrad_ref

    def montecarlo_average_var(self, mean, cov, n_samples=100000, func=None):
        """Computes the variance and expected value using Monte Carlo approach.

        :param mean: the mean value.
        :param cov: the covariance matrix.
        :param n_samples: the number of samples for the distribution
            (Default value = 100000).
        :param func: if None, the compute_function_approximation function,
                        otherwise a user function (Default value = None).
        """
        n_dv = len(mean)
        if not cov.shape == (n_dv, n_dv):
            raise ValueError(
                "Covariance matrix dimension " + "incompatible with mean dimensions"
            )
        ran = multivariate_normal(mean=mean, cov=cov, size=n_samples).T
        vals = np.zeros(n_samples)
        if func is None:
            func = self.compute_function_approximation
        for i in range(n_samples):
            vals[i] = func(ran[:, i])
        average = np.average(vals)
        var = np.var(vals)
        return average, var
