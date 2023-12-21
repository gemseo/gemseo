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

from typing import TYPE_CHECKING
from typing import Callable
from typing import Final

import numpy as np
from numpy.random import default_rng
from strenum import StrEnum

from gemseo import SEED
from gemseo.post.core.hessians import BFGSApprox
from gemseo.post.core.hessians import LSTSQApprox
from gemseo.post.core.hessians import SR1Approx

if TYPE_CHECKING:
    from collections.abc import Sized


class RobustnessQuantifier:
    """classdocs."""

    class Approximation(StrEnum):
        """The approximation types."""

        BFGS = "BFGS"
        SR1 = "SR1"
        LEAST_SQUARES = "LEAST_SQUARES"

    __APPROXIMATION_TO_METHOD: Final[Approximation, Callable] = {
        Approximation.BFGS: BFGSApprox,
        Approximation.SR1: SR1Approx,
        Approximation.LEAST_SQUARES: LSTSQApprox,
    }

    def __init__(
        self, history, approximation_method: Approximation = Approximation.SR1
    ) -> None:
        """
        Args:
            history: An approximation history.
            approximation_method: The approximation method for the Hessian.
        """  # noqa: D205, D212, D415
        self.history = history
        self.approximator = self.__APPROXIMATION_TO_METHOD[approximation_method](
            history
        )
        self.b_mat = None
        self.x_ref = None
        self.f_ref = None
        self.fgrad_ref = None

    def compute_approximation(
        self,
        funcname: str,
        first_iter: int = 0,
        last_iter: int | None = None,
        b0_mat=None,
        at_most_niter: int | None = None,
        func_index=None,
    ):
        """Build the BFGS approximation for the Hessian.

        Args:
            funcname: The name of the function.
            first_iter: The index of the first iteration.
            last_iter: The last iteration of the history to be considered.
                If ``None``, consider all the iterations.
            b0_mat: The Hessian matrix at the first iteration.
            at_most_niter: The maximum number of iterations to be considered.
                If ``None``, consider all the iterations.
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

    def compute_expected_value(self, expect: Sized, cov):
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
        exp_val = np.trace(b_approx @ cov)
        delta = expect - self.x_ref
        exp_val += delta.T @ (b_approx @ delta)
        return exp_val

    def compute_variance(self, expect: Sized, cov):
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
        b_approx_cov = b_approx @ cov
        v_mat = b_approx_cov @ b_approx_cov
        b_approx_mu_cent = b_approx @ mu_cent
        v_mat += 4 * b_approx_mu_cent.T @ (cov @ b_approx_mu_cent)
        return 2 * np.trace(v_mat)

    def compute_function_approximation(self, x_vars) -> float:
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
        return 0.5 * x_l.T @ (self.b_mat @ x_l) + self.fgrad_ref.T @ x_l + self.f_ref

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
        return self.b_mat @ x_l + self.fgrad_ref

    def montecarlo_average_var(
        self, mean: Sized, cov, n_samples: int = 100000, func=None
    ):
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
        ran = default_rng(SEED).multivariate_normal(mean, cov, n_samples).T
        vals = np.zeros(n_samples)
        if func is None:
            func = self.compute_function_approximation
        for i in range(n_samples):
            vals[i] = func(ran[:, i])
        average = np.average(vals)
        var = np.var(vals)
        return average, var
