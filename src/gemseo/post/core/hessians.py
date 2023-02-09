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
r"""Approximation of the Hessian matrix from an optimization history.

Notations:

- :math:`f`: the function of interest for which to approximate the Hessian matrix,
- :math:`y`: the output value of :math:`f`,
- :math:`x\in\mathbb{R}^d`: the :math:`d` input variables of :math:`f`,
- :math:`k`: the :math:`k`-th iteration of the optimization history,
- :math:`K`: the iteration of the optimization history
  at which to approximate the Hessian matrix,
- :math:`x_k`: the input value at iteration :math:`k`,
- :math:`\Delta x_k=x_{k+1}-x_k`: the variation of :math:`x`
  from iteration :math:`k` to iteration :math:`k+1`,
- :math:`y_k`: the output value at iteration :math:`k`,
- :math:`\Delta y_k=y_{k+1}-y_k`: the variation of the function output
  from iteration :math:`k` to iteration :math:`k+1`,
- :math:`g_k`: the gradient of :math:`f` at :math:`x_k`,
- :math:`\Delta g_k=g_{k+1}-g_k`: the variation of the gradient
  from iteration :math:`k` to iteration :math:`k+1`,
- :math:`B_k`: the approximation of the Hessian of :math:`f` at :math:`x_k`,
- :math:`H_k`: the inverse of :math:`B_k`.
"""
from __future__ import annotations

import logging
from typing import Generator

from docstring_inheritance import GoogleDocstringInheritanceMeta
from numpy import array
from numpy import atleast_2d
from numpy import concatenate
from numpy import cumsum
from numpy import diag as np_diag
from numpy import dot
from numpy import eye
from numpy import inf
from numpy import ndarray
from numpy import newaxis
from numpy import sqrt
from numpy import trace
from numpy import zeros
from numpy.linalg import cholesky
from numpy.linalg import inv
from numpy.linalg import LinAlgError
from numpy.linalg import multi_dot
from numpy.linalg import norm
from numpy.matlib import repmat
from scipy.optimize import leastsq

from gemseo.algos.database import Database
from gemseo.algos.design_space import DesignSpace

LOGGER = logging.getLogger(__name__)


class HessianApproximation(metaclass=GoogleDocstringInheritanceMeta):
    r"""Approximation of the Hessian matrix from an optimization history."""

    history: Database
    """The optimization history containing input values, output values and Jacobian
    values."""

    x_ref: ndarray | None
    """The value :math:`x_K` of the input variables :math:`x` at the iteration :math:`K`
    of the optimization history; this is the point at which the Hessian matrix and its
    inverse are approximated."""

    fgrad_ref: ndarray | None
    """The value :math:`g_K` of the gradient function :math:`g` of :math:`f` at
    :math:`x_K`."""

    f_ref: ndarray | None
    """The value :math:`y_K` of the output of :math:`f` at :math:`x_K`."""

    b_mat_history: list[ndarray]
    r"""The history :math:`B_0,B_1,\ldots,B_K` of the approximations of the Hessian
    matrix :math:`B`."""

    h_mat_history: list[ndarray]
    r"""The history :math:`H_0,H_1,\ldots,H_K` of the approximations of the inverse
    Hessian matrix :math:`H`."""

    def __init__(
        self,
        history: Database,
    ) -> None:
        """
        Args:
            history: The optimization history
                containing input values, output values and Jacobian values.
        """  # noqa: D205, D212, D415
        self.history = history
        self.x_ref = None
        self.fgrad_ref = None
        self.f_ref = None
        self.b_mat_history = []
        self.h_mat_history = []

    def get_x_grad_history(
        self,
        funcname: str,
        first_iter: int = 0,
        last_iter: int = 0,
        at_most_niter: int = -1,
        func_index: int | None = None,
        normalize_design_space: bool = False,
        design_space: DesignSpace | None = None,
    ) -> tuple[ndarray, ndarray, int, int]:
        """Return the histories of the inputs and gradient.

        Args:
            funcname: The name of the function for which to get the gradient.
            first_iter: The first iteration of the history to be considered.
            last_iter: The last iteration of the history to be considered.
                If 0, consider all the iterations.
            at_most_niter: The maximum number of iterations to be considered.
                If -1, consider all the iterations.
            func_index: The index of the output of interest
                to be defined if the function has a multidimensional output.
                If ``None`` and if the output is multidimensional, an error is raised.
            normalize_design_space: Whether to scale the input values between 0 and 1
                to work in a normalized input space.
            design_space: The input space used to scale the input values
                if ``normalize_design_space`` is ``True``.

        Returns:
            * The history of the input variables.
            * The history of the gradient.
            * The length of the history.
            * The dimension of the input space.

        Raises:
            ValueError: When either
                the gradient history contains a single element,
                ``func_index`` is ``None`` while the function output is a vector,
                ``func_index`` is not an output index,
                the shape of the history of the input variables
                is not consistent with the shape of the history of the gradient
                or the optimization history size is insufficient.
        """
        # TODO: use None as default value for last_iter and at_most_iter
        grad_hist, x_hist = self.history.get_func_grad_history(funcname, x_hist=True)
        if normalize_design_space:
            (
                x_hist,
                grad_hist,
            ) = self._normalize_x_g(x_hist, grad_hist, design_space)

        grad_hist_length = len(grad_hist)
        assert grad_hist_length == len(x_hist)  # TODO: remove it

        if grad_hist_length < 2:
            raise ValueError(
                f"Cannot build approximation for function: {funcname} "
                f"because its gradient history is too small: {grad_hist_length}."
            )

        grad_hist = array(grad_hist)
        if grad_hist.ndim == 1:
            grad_hist = grad_hist[:, newaxis]

        x_hist = array(x_hist)
        if x_hist.shape != (grad_hist.shape[0], grad_hist.shape[-1]):
            # TODO: add shapes in the exception message
            raise ValueError(
                "Inconsistent gradient and design variables optimization history."
            )

        # Function is a vector, Jacobian is a 2D matrix
        if grad_hist.ndim == 3:
            if grad_hist.shape[1] == 1:
                func_index = 0
            else:
                if func_index is None:
                    raise ValueError(
                        f"Function {funcname} has a vector output, "
                        "the function index of the output must be specified."
                    )

                output_size = grad_hist.shape[1]
                if not 0 <= func_index < output_size:
                    raise ValueError(
                        f"Function {funcname} has a vector output "
                        f"of size {output_size}, "
                        f"function index {func_index} is out of range."
                    )

            grad_hist = grad_hist[:, func_index, :]

        if last_iter == 0:
            x_hist = x_hist[first_iter:, :]
            grad_hist = grad_hist[first_iter:, :]
        else:
            x_hist = x_hist[first_iter:last_iter, :]
            grad_hist = grad_hist[first_iter:last_iter, :]

        n_iterations = x_hist.shape[0]
        if 0 < at_most_niter < n_iterations:
            x_hist = x_hist[n_iterations - at_most_niter :, :]
            grad_hist = grad_hist[n_iterations - at_most_niter :, :]

        n_iterations, input_dimension = x_hist.shape
        if n_iterations < 2 or input_dimension == 0:
            # TODO: split into two tests
            raise ValueError(
                "Insufficient optimization history size, "
                f"niter={n_iterations} nparam={input_dimension}."
            )

        self.x_ref = x_hist[-1]
        self.fgrad_ref = grad_hist[-1]
        if last_iter == 0:
            self.f_ref = array(self.history.get_func_history(funcname))[-1]
        else:
            self.f_ref = array(self.history.get_func_history(funcname))[:last_iter][-1]

        return x_hist, grad_hist, n_iterations, input_dimension

    @staticmethod
    def _normalize_x_g(
        x_hist: ndarray,
        x_grad_hist: ndarray,
        design_space: DesignSpace,
    ) -> tuple[ndarray, ndarray]:
        """Scale the design variables between 0 and 1 in the histories.

        Args:
            x_hist: The history of the input variables.
            x_grad_hist: The history of the gradient.
            design_space: The input space used to scale the input variables.

        Returns:
            * The history of the scaled input variables.
            * The history of the gradient.

        Raises:
            ValueError: When the input space is ``None``.
        """
        if design_space is None:
            raise ValueError(
                "Design space must be provided "
                "when using a normalize_design_space option."
            )

        scaled_x_hist, scaled_grad_hist = [], []
        for x_value, grad_value in zip(x_hist, x_grad_hist):
            scaled_x_hist.append(design_space.normalize_vect(x_value))
            scaled_grad_hist.append(design_space.normalize_grad(grad_value))

        return scaled_x_hist, scaled_grad_hist

    @staticmethod
    def get_s_k_y_k(
        x_hist: ndarray,
        x_grad_hist: ndarray,
        iteration: int,
    ) -> tuple[ndarray, ndarray]:
        r"""Compute the variation of the input variables and gradient from an iteration.

        The variations from the iteration :math:`k` are defined by:

        - :math:`\Delta x_k = x_{k+1}-x_k` for the input variables,
        - :math:`\Delta g_k = g_{k+1} - g_k` for the gradient.

        Args:
            x_hist: The history of the input variables.
            x_grad_hist: The history of the gradient.
            iteration: The optimization iteration at which to compute the variations.

        Returns:
            * The difference between the input variables at iteration ``iteration+1``
              and the input variables at iteration ``iteration``.
            * The difference between the gradient at iteration ``iteration+1``
              and the gradient at iteration ``iteration``.

        Raises:
            ValueError: When the iteration is not stored in the database.
        """
        n_iterations = x_grad_hist.shape[0]
        if iteration >= n_iterations:
            raise ValueError(
                f"Iteration {iteration} is higher than the number of gradients "
                f"in the database: {n_iterations}."
            )

        input_diff = atleast_2d(x_hist[iteration + 1] - x_hist[iteration]).T
        grad_diff = atleast_2d(x_grad_hist[iteration + 1] - x_grad_hist[iteration]).T
        return input_diff, grad_diff

    @staticmethod
    def iterate_s_k_y_k(
        x_hist: ndarray,
        x_grad_hist: ndarray,
    ) -> Generator[tuple[ndarray, ndarray]]:
        r"""Compute the variations of the input variables and gradient.

        The variations from the iteration :math:`k` are defined by:

        - :math:`\Delta x_k = x_{k+1}-x_k` for the input variables,
        - :math:`\Delta g_k = g_{k+1} - g_k` for the gradient.

        Args:
            x_hist: The history of the input variables.
            x_grad_hist: The history of the gradient.

        Returns:
            * The difference between the input variables at iteration ``iteration``
              and the input variables at iteration ``iteration+1``.
            * The difference between the gradient at iteration ``iteration``
              and the gradient at iteration ``iteration+1``.
        """
        for iteration in range(len(x_hist) - 1):
            input_diff, grad_diff = HessianApproximation.get_s_k_y_k(
                x_hist, x_grad_hist, iteration
            )
            yield input_diff, grad_diff

    def build_approximation(
        self,
        funcname: str,
        save_diag: bool = False,
        first_iter: int = 0,
        last_iter: int = -1,
        b_mat0: ndarray | None = None,
        at_most_niter: int = -1,
        return_x_grad: bool = False,
        func_index: int | None = None,
        save_matrix: bool = False,
        scaling: bool = False,
        normalize_design_space: bool = False,
        design_space: DesignSpace | None = None,
    ) -> tuple[ndarray, ndarray, ndarray | None, ndarray | None]:
        # pylint: disable=W0221
        """Compute :math:`B`, the approximation of the Hessian matrix.

        Args:
            funcname: The name of the function
                for which to approximate the Hessian matrix.
            save_diag: Whether to return the approximations of the Hessian's diagonal.
            first_iter: The first iteration of the history to be considered.
            last_iter: The last iteration of the history to be considered.
            b_mat0: The initial approximation of the Hessian matrix.
            at_most_niter: The maximum number of iterations to be considered.
            return_x_grad: Whether to return the input variables and gradient
                at the last iteration.
            func_index: The index of the output of interest
                to be defined if the function has a multidimensional output.
                If ``None`` and if the output is multidimensional, an error is raised.
            save_matrix: Whether to store the approximations of the Hessian
                in :attr:`.HessianApproximation.b_mat_history`.
            scaling: do scaling step
            normalize_design_space: Whether to scale the input values between 0 and 1
                to work in a normalized input space.
            design_space: The input space used to scale the input values
                if ``normalize_design_space`` is ``True``.

        Returns:
            * :math:`B`, the approximation of the Hessian matrix.
            * The diagonal of :math:`B`.
            * The history of the input variables if ``return_x_grad`` is ``True``.
            * The history of the gradient if ``return_x_grad`` is ``True``.
        """
        x_hist, grad_hist, _, _ = self.get_x_grad_history(
            funcname,
            first_iter,
            last_iter,
            at_most_niter,
            func_index,
            normalize_design_space,
            design_space,
        )
        if b_mat0 is None:
            last_n_grad = grad_hist.shape[0] - 2
            input_diff, grad_diff = self.get_s_k_y_k(x_hist, grad_hist, last_n_grad)
            alpha = dot(grad_diff.T, input_diff) / dot(grad_diff.T, grad_diff)
            hessian = (1.0 / alpha) * eye(grad_hist.shape[1])
        elif b_mat0.size == 0:
            hessian = zeros((x_hist.shape[1],) * 2)
        else:
            hessian = b_mat0

        hessian_diagonal = []

        for input_diff, grad_diff in self.iterate_s_k_y_k(x_hist, grad_hist):
            self.iterate_approximation(hessian, input_diff, grad_diff, scaling=scaling)
            if save_diag:
                hessian_diagonal.append(np_diag(hessian).copy())
            if save_matrix:
                self.b_mat_history.append(hessian.copy())

        if return_x_grad:
            return hessian, hessian_diagonal, x_hist[-1, :], grad_hist[-1, :]

        return hessian, hessian_diagonal, None, None

    @staticmethod
    def compute_scaling(
        hessk: ndarray,
        hessk_dsk: ndarray,
        dskt_hessk_dsk: ndarray,
        dyk: ndarray,
        dyt_dsk: ndarray,
    ) -> tuple[float, float]:
        r"""Compute the scaling coefficients :math:`c_1` and :math:`c_2`.

        - :math:`c_1=\frac{d-1}{\mathrm{Tr}(B_k)-\frac{\|B_k\Delta x_k\|_2^2}
          {\Delta x_k^T B_k\Delta x_k}}`,
        - :math:`c_2=\frac{\Delta g_k^T\Delta x_k}{\|\Delta g_k\|_2^2}`.

        Args:
            hessk: The approximation :math:`B_k` of the Hessian matrix
                at iteration :math:`k`.
            hessk_dsk: The product :math:`B_k\Delta x_k`.
            dskt_hessk_dsk: The product :math:`\Delta x_k^T B_k\Delta x_k`.
            dyk: The variation of the gradient :math:`\Delta g_k`.
            dyt_dsk: The product
                :math:`\Delta g_k^T\Delta x_k`.

        Returns:
            * coeff1: TODO
            * coeff2: TODO
        """
        coeff1 = (len(hessk_dsk) - 1) / (
            trace(hessk) - norm(hessk_dsk) ** 2 / dskt_hessk_dsk
        )
        coeff2 = dyt_dsk / norm(dyk) ** 2
        return coeff1, coeff2

    @staticmethod
    def iterate_approximation(
        hessk: ndarray,
        dsk: ndarray,
        dyk: ndarray,
        scaling: bool = False,
    ) -> None:
        r"""Update :math:`B` from iteration :math:`k` to iteration :math:`k+1`.

        Based on an iteration of the BFGS algorithm:

        :math:`B_{k+1} =
        B_k
        - c_1\frac{B_k\Delta x_k\Delta x_k^TB_k}{\Delta x_k^TB_k\Delta x_k}
        + c_2\frac{\Delta g_k\Delta g_k^T}{\Delta g_k^T\Delta x_k}`

        where :math:`c_1=c_2=1` if ``scaling`` is ``False``, otherwise:

        - :math:`c_1=\frac{d-1}{\mathrm{Tr}(B_k)-\frac{\|B_k\Delta x_k\|_2^2}
          {\Delta x_k^T B_k\Delta x_k}}`,
        - :math:`c_2=\frac{\Delta g_k^T\Delta x_k}{\|\Delta g_k\|_2^2}`.

        .. note::
            ``hessk`` represents :math:`B_k` initially
            before to be overwritten by :math:`B_{k+1}` when passed to this method.

        .. seealso::
            `BFGS algorithm.
            <https://en.wikipedia.org/wiki/Broyden-Fletcher-Goldfarb-Shanno_algorithm>`_

        Args:
            hessk: The approximation :math:`B_k` of the Hessian matrix
                at iteration :math:`k`.
            dsk: The variation :math:`\Delta x_k` of the input variables.
            dyk: The variation :math:`\Delta g_k` of the gradient.
            scaling: Whether to use a scaling stage.
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
        funcname: str,
        save_diag: int = False,
        first_iter: int = 0,
        last_iter: int = -1,
        h_mat0: ndarray | None = None,
        at_most_niter: int = -1,
        return_x_grad: bool = False,
        func_index: int | None = None,
        save_matrix: bool = False,
        factorize: bool = False,
        scaling: bool = False,
        angle_tol: float = 1e-5,
        step_tol: float = 1e10,
        normalize_design_space: bool = False,
        design_space: DesignSpace | None = None,
    ) -> tuple[ndarray, ndarray, ndarray | None, ndarray | None]:
        r"""Compute :math:`H`, the approximation of the inverse of the Hessian matrix.

        Args:
            funcname: The name of the function
                for which to approximate the inverse of the Hessian matrix.
            save_diag: Whether to return the list of diagonal approximations.
            first_iter: The first iteration of the history to be considered.
            last_iter: The last iteration of the history to be considered.
            h_mat0: The initial approximation of the inverse of the Hessian matrix.
                If None,
                use :math:`H_0=\frac{\Delta g_k^T\Delta x_k}
                {\Delta g_k^T\Delta g_k}I_d`.
            at_most_niter: The maximum number of iterations to take.
            return_x_grad: Whether to return the input variables and gradient
                at the last iteration.
            func_index: The output index of the function
                to be provided if the function output is a vector.
            save_matrix: Whether to store the approximations of the inverse Hessian
                in :attr:`.HessianApproximation.h_mat_history`.
            factorize: Whether to factorize the approximations of the Hessian matrix
                and its inverse, as :math:`A=A_{1/2}A_{1/2}^T` for a matrix :math:`A`.
            scaling: do scaling step
            angle_tol: The significativity level for
                :math:`\Delta g_k^T\Delta x_k`.
            step_tol: The significativity level for
                :math:`\|\Delta g_k\|_{\infty}`.
            normalize_design_space: Whether to scale the input values between 0 and 1
                to work in a normalized input space.
            design_space: The input space used to scale the input values
                if ``normalize_design_space`` is ``True``.

        Returns:
            * :math:`H`, the approximation of the inverse of the Hessian matrix.
            * The diagonal of :math:`H`.
            * The history of the input variables if ``return_x_grad`` is ``True``.
            * The history of the gradient if ``return_x_grad`` is ``True``.
            * The matrix :math:`H_{1/2}` such that :math:`H=H_{1/2}H_{1/2}^T`
              if ``factorize`` is ``True``.
            * :math:`B`, the approximation of the Hessian matrix.
            * A matrix :math:`B_{1/2}` such that :math:`B=B_{1/2}B_{1/2}^T`
              if ``factorize`` is ``True``.

        Raises:
            LinAlgError: When either
                the inversion of :math:`H` fails
                or the Cholesky decomposition of :math:`H` or :math:`B` fails.
        """
        x_hist, grad_hist, _, _ = self.get_x_grad_history(
            funcname,
            first_iter,
            last_iter,
            at_most_niter,
            func_index,
            normalize_design_space,
            design_space,
        )
        h_factor = None  # to become a matrix G such that H = G*G', optionally
        b_factor = None  # to become the inverse of the matrix G
        if h_mat0 is None:
            last_n_grad = grad_hist.shape[0] - 2
            s_k, y_k = self.get_s_k_y_k(x_hist, grad_hist, last_n_grad)
            alpha = dot(y_k.T, s_k) / dot(y_k.T, y_k)
            n_x = len(grad_hist[0])
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
                raise LinAlgError("The inversion of h_mat failed.")

            if factorize or scaling:
                try:
                    h_factor = cholesky(h_mat)
                    b_factor = cholesky(b_mat).T
                except LinAlgError:
                    raise LinAlgError(
                        "The Cholesky decomposition of h_factor or b_factor failed."
                    )

        diag = []
        count = 0
        k = 0
        for s_k, y_k in self.iterate_s_k_y_k(x_hist, grad_hist):
            k += 1
            if dot(s_k.T, y_k) > angle_tol and norm(y_k, inf) < step_tol:
                count += 1
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
            return h_mat, diag, x_hist[-1, :], grad_hist[-1, :], None, None, None

        return h_mat, diag, None, None, h_factor, b_mat, b_factor

    @staticmethod
    def compute_corrections(
        x_hist: ndarray,
        x_grad_hist: ndarray,
    ) -> tuple[ndarray, ndarray]:
        """Compute the successive variations of both input variables and gradient.

        These variations are called *corrections*.

        Args:
            x_hist: The history of the input variables.
            x_grad_hist: The history of the gradient.

        Returns:
            * The successive variations of the input variables.
            * The successive variations of the gradient.
        """
        n_iter = len(x_hist)
        x_corr = x_hist[1:n_iter].T - x_hist[: n_iter - 1].T
        grad_corr = x_grad_hist[1:n_iter].T - x_grad_hist[: n_iter - 1].T
        return x_corr, grad_corr

    @staticmethod
    def rebuild_history(
        x_corr: ndarray,
        x_0: ndarray,
        grad_corr: ndarray,
        g_0: ndarray,
    ) -> tuple[ndarray, ndarray]:
        """Compute the history from the corrections of input variables and gradient.

        A *correction* is the variation of a quantity between two successive iterations.

        Args:
            x_corr: The corrections of the input variables.
            x_0: The initial values of the input variables.
            grad_corr: The corrections of the gradient.
            g_0: The initial value of the gradient.

        Returns:
            * The history of the input variables.
            * The history of the gradient.
        """
        # Rebuild the argument history:
        x_hist = repmat(x_0, x_corr.shape[1], 1) + cumsum(x_corr.T, axis=0)
        x_hist = concatenate((atleast_2d(x_0), x_hist), axis=0)
        # Rebuild the gradient history
        x_grad_hist = repmat(g_0, grad_corr.shape[1], 1) + cumsum(grad_corr.T, axis=0)
        x_grad_hist = concatenate((atleast_2d(g_0), x_grad_hist), axis=0)
        return x_hist, x_grad_hist

    @staticmethod
    def iterate_inverse_approximation(
        h_mat: ndarray,
        s_k: ndarray,
        y_k: ndarray,
        h_factor: ndarray | None = None,
        b_mat: ndarray | None = None,
        b_factor: ndarray | None = None,
        factorize: bool = False,
        scaling: bool = False,
    ):
        r"""Update :math:`H` and :math:`B` from step :math:`k` to step :math:`k+1`.

        Use an iteration of the BFGS algorithm:

        :math:`B_{k+1} =
        B_k
        - c_1\frac{B_k\Delta x_k\Delta x_k^TB_k}{\Delta x_k^TB_k\Delta x_k}
        + c_2\frac{\Delta g_k\Delta g_k^T}{\Delta g_k^T\Delta x_k}`

        and

        :math:`H_{k+1}=c_1^{-1}\Pi_{k+1}H_k\Pi_{k+1}^T
        +c_2^{-1}\frac{\Delta x_k\Delta x_k^T}{\Delta g_k^T\Delta x_k}`

        where:

        :math:`\Pi_{k+1}=I_d-\frac{\Delta x_k\Delta g_k^T}
        {\Delta g_k^T\Delta x_k}`

        and where :math:`c_1=c_2=1` if ``scaling`` is ``False``, otherwise:

        - :math:`c_1=\frac{d-1}{\mathrm{Tr}(B_k)-\frac{\|B_k\Delta x_k\|_2^2}
          {\Delta x_k^T B_k\Delta x_k}}`,
        - :math:`c_2=\frac{\Delta g_k^T\Delta x_k}{\|\Delta g_k\|_2^2}`.

        .. note::
            ``h_mat`` and ``b_mat`` represent :math:`H_k` and :math:`B_k` initially
            before to be overwritten by :math:`H_{k+1}` and :math:`B_{k+1}`
            when passed to this method.

        .. seealso::
            `BFGS algorithm.
            <https://en.wikipedia.org/wiki/Broyden-Fletcher-Goldfarb-Shanno_algorithm>`_

        Args:
            h_mat: The approximation :math:`H_k` of the inverse of the Hessian matrix
                at iteration :math:`k`.
            s_k: The variation :math:`\Delta x_k` of the input variables.
            y_k: The variation :math:`\Delta g_k` of the gradient.
            h_factor: The square root of the :math:`H_k` at iteration :math:`k`.
            b_mat: The approximation :math:`B_k` of the Hessian matrix
                at iteration :math:`k` if ``factorize`` is ``True``.
            b_factor: The square root of the :math:`B_k` at iteration :math:`k`
                if ``factorize`` is ``True``.
            factorize: Whether to update the approximations of the Hessian matrix
                and its inverse, as :math:`A=A_{1/2}A_{1/2}^T` for a matrix :math:`A`.
            scaling: do scaling step
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
            # b_factor[:, :] = dot(eye(len(s_k)) - sstB.T / stBs / sqrt(coeff1)
            #                      + dot(y_k, s_k.T)
            #                      / sqrt(coeff2 * stBs * yts),
            #                      b_factor)
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
    """Hessian matrix approximation with the BFGS algorithm."""

    @staticmethod
    def iterate_s_k_y_k(  # noqa:D102
        x_hist: ndarray,
        x_grad_hist: ndarray,
    ) -> Generator[tuple[ndarray, ndarray]]:
        for iteration in range(len(x_hist) - 1):
            input_diff, grad_diff = BFGSApprox.get_s_k_y_k(
                x_hist, x_grad_hist, iteration
            )
            # All pairs curvatures shall be positive
            # if dot(s_k.T, y_k) > 0.:
            if dot(input_diff.T, grad_diff) > 1e-16 * dot(grad_diff.T, grad_diff):
                yield input_diff, grad_diff


class SR1Approx(HessianApproximation):
    r"""Hessian matrix approximation with the Symmetric Rank One (SR1) algorithm.

    The approximation at iteration :math:`k+1` is:

    .. math::
       B_{k+1}=B_k +
       \frac{(\Delta g_k-B_k\Delta x_k)(\Delta g_k-B_k\Delta x_k)^T}
       {(\Delta g_k-B_k\Delta x_k)^T\Delta x_k}

    This update from iteration :math:`k` to iteration :math:`k+1` is applied only if
    :math:`|(\Delta g_k-B_k\Delta x_k)^T\Delta x_k|
    \geq \varepsilon\|\Delta x_k\|\|\Delta g_k\|`
    where :math:`\varepsilon` is a small number, e.g. :math:`10^{-8}`.

    .. seealso::

       `SR1 algorithm. <https://en.wikipedia.org/wiki/Symmetric_rank-one>`_
    """

    EPSILON = 1e-8

    @staticmethod
    def iterate_approximation(  # noqa:D102
        b_mat: ndarray,
        s_k: ndarray,
        y_k: ndarray,
        scaling: bool = False,
    ):
        residuals = y_k - multi_dot((b_mat, s_k))
        denominator = multi_dot((residuals.T, s_k))
        if abs(denominator) > SR1Approx.EPSILON * norm(s_k) * norm(residuals):
            b_mat[:, :] = b_mat + multi_dot((residuals, residuals.T)) / denominator
        else:
            LOGGER.debug(
                "Denominator of SR1 update is too small, update skipped %s.",
                denominator,
            )


class LSTSQApprox(HessianApproximation):
    """Least squares approximation of a Hessian matrix from an optimization history."""

    def build_approximation(  # noqa:D102
        self,
        funcname: str,
        save_diag: bool = False,
        first_iter: int = 0,
        last_iter: int = -1,
        b_mat0: ndarray | None = None,
        at_most_niter: int = -1,
        return_x_grad: bool = False,
        scaling: bool = False,
        func_index: int = -1,
        normalize_design_space: bool = False,
        design_space: DesignSpace | None = None,
    ) -> tuple[ndarray, ndarray, ndarray | None, ndarray | None]:
        x_hist, grad_hist, n_iterations, input_dimension = self.get_x_grad_history(
            funcname,
            first_iter,
            last_iter,
            at_most_niter,
            func_index=func_index,
            normalize_design_space=normalize_design_space,
            design_space=design_space,
        )
        assert len(grad_hist) == len(x_hist)  # TODO: replace with an if/raise

        sec_dim = max(input_dimension, n_iterations)
        hessian_diagonal = []

        def y_to_b(
            y_vars: ndarray,
        ) -> ndarray:
            """Reshape the approximation from vector to matrix.

            Args:
                y_vars: The vector approximation.

            Returns:
                The square matrix version of the passed vector.
            """
            y_mat = y_vars.reshape((input_dimension, input_dimension))
            return y_mat + y_mat.T

        def compute_error(
            y_vars: ndarray,
        ) -> ndarray:
            """Create the least square function.

            Args:
                y_vars: The current approximation vector.

            Returns:
                The estimated error vector.
            """
            hessian = y_to_b(y_vars)
            err = zeros((input_dimension, sec_dim))
            for item, x_current in enumerate(x_hist):
                err[:, item] = dot(hessian, x_current - self.x_ref) - grad_hist[item]

            err = err.reshape(-1)
            if n_iterations < input_dimension:
                err += y_vars

            return err

        x_0 = zeros(input_dimension * input_dimension)
        LOGGER.debug("Start least squares problem..")
        x_opt, ier = leastsq(compute_error, x0=x_0)  # , cov_x, infodict, mesg, ier
        LOGGER.debug("End least squares, msg=%s", str(ier))
        hessian = y_to_b(x_opt)

        if return_x_grad:
            return hessian, hessian_diagonal, x_hist[-1, :], grad_hist[-1, :]

        return hessian, hessian_diagonal, None, None
