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
from __future__ import annotations

import numpy as np
from numpy.random import multivariate_normal

from gemseo.post.core.hessians import BFGSApprox
from gemseo.post.core.hessians import LSTSQApprox
from gemseo.post.core.hessians import SR1Approx


class RobustnessQuantifier:
    """classdocs."""

    AVAILABLE_APPROXIMATIONS = ["BFGS", "SR1", "LEAST_SQUARES"]

    def __init__(self, history, approximation_method="SR1"):
        """
        Args:
            history: An approximation history.
            approximation_method: The name of an approximation method for the Hessian.
        """  # noqa: D205, D212, D415
        self.history = history
        if approximation_method not in self.AVAILABLE_APPROXIMATIONS:
            raise ValueError(
                f"Unknown hessian approximation method {approximation_method}; "
                f"the available ones are: {self.AVAILABLE_APPROXIMATIONS}."
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
        """Build the BFGS approximation for the Hessian.

        Args:
            funcname: The name of the function.
            first_iter: The index of the first iteration.
            last_iter: The index of the last iteration.
            b0_mat: The Hessian matrix at the first iteration.
            at_most_niter: The maximum number of iterations to take
            func_index: The component of the function.

        Returns:
            An approximation of the Hessian matrix.
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
        r"""Compute the expected value of the output.

        Equal to :math:`0.5\mathbb{E}[e^TBe]`
        where :math:`e` is the expected values
        and :math:`B` the covariance matrix.

        Args:
            expect: The expected value of the inputs.
            cov: The covariance matrix of the inputs.

        Returns:
            The expected value of the output.

        Raises:
            ValueError: When expectation and covariance matrices
                have inconsistent shapes or when the Hessian approximation is missing.
        """
        n_vars = len(expect)
        if cov.shape != (n_vars, n_vars):
            raise ValueError("Inconsistent expect and covariance matrices shapes")
        if self.b_mat is None:
            raise ValueError(
                "Build Hessian approximation before computing expected_value_offset"
            )
        b_approx = 0.5 * self.b_mat
        exp_val = np.trace(np.dot(b_approx, cov))
        exp_val += np.linalg.multi_dot(
            ((expect - self.x_ref).T, b_approx, (expect - self.x_ref))
        )
        return exp_val

    def compute_variance(self, expect, cov):
        r"""Compute the variance of the output.

        Equal to :math:`0.5\mathbb{E}[e^TBe]`
        where :math:`e` is the expected values
        and :math:`B` the covariance matrix.

        Args:
            expect: The expected value of the inputs.
            cov: The covariance matrix of the inputs.

        Returns:
            The variance of the output.

        Raises:
            ValueError: When expectation and covariance matrices
                have inconsistent shapes or when the Hessian approximation is missing.
        """
        if self.b_mat is None:
            raise ValueError("Build Hessian approximation before computing variance")
        n_vars = len(expect)
        if cov.shape != (n_vars, n_vars):
            raise ValueError("Inconsistent expect and covariance matrices shapes")
        b_approx = 0.5 * self.b_mat
        mu_cent = expect - self.x_ref
        v_mat = np.linalg.multi_dot((b_approx, cov, b_approx, cov))
        v_mat += 4 * np.linalg.multi_dot((mu_cent.T, b_approx, cov, b_approx, mu_cent))
        return 2 * np.trace(v_mat)

    def compute_function_approximation(self, x_vars):
        """Compute a second order approximation of the function.

        Args:
            x_vars: The point on which the approximation is evaluated.

        Returns:
            A second order approximation of the function.
        """
        if self.b_mat is None or self.x_ref is None:
            raise ValueError(
                "Build Hessian approximation before computing function approximation"
            )
        x_l = x_vars - self.x_ref
        return (
            0.5 * np.linalg.multi_dot((x_l.T, self.b_mat, x_l))
            + np.dot(self.fgrad_ref.T, x_l)
            + self.f_ref
        )

    def compute_gradient_approximation(self, x_vars):
        """Computes a first order approximation of the gradient based on the hessian.

        Args:
            x_vars: The point on which the approximation is evaluated.
        """
        if self.b_mat is None or self.fgrad_ref is None:
            raise ValueError(
                "Build Hessian approximation before computing function approximation"
            )
        x_l = x_vars - self.x_ref
        return np.dot(self.b_mat, x_l) + self.fgrad_ref

    def montecarlo_average_var(self, mean, cov, n_samples=100000, func=None):
        """Computes the variance and expected value using Monte Carlo approach.

        Args:
            mean: The mean value.
            cov: The covariance matrix.
            n_samples: The number of samples for the distribution.
            func: If ``None``, the ``compute_function_approximation`` function,
                otherwise a user function.
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
