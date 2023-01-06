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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Linear equations problem."""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from numpy import ndarray
from numpy.linalg import norm
from scipy.sparse import spmatrix
from scipy.sparse.linalg import LinearOperator


class LinearProblem:
    """Representation of the linear equations' system ``A.x = b``.

    It also contains the solution, and some properties of the system such as the symmetry
    or positive definiteness.
    """

    rhs: ndarray
    """The right-hand side of the equation."""

    lhs: ndarray | LinearOperator | spmatrix
    """The left-hand side of the equation.

    If None, the problem can't be solved and the user has to set it after init.
    """

    solution: ndarray
    """The current solution of the problem."""

    is_converged: bool
    """If the solution is_converged."""

    convergence_info: int | str
    """The information provided by the solver if convergence occurred or not."""

    is_symmetric: bool
    """Whether the LHS is symmetric."""

    is_positive_def: bool
    """Whether the LHS is positive definite."""

    is_lhs_linear_operator: bool
    """Whether the LHS is symmetric."""

    solver_options: dict[str, Any]
    """The options passed to the solver."""

    solver_name: str
    """The solver name."""

    residuals_history: list[float]
    """The convergence history of residuals."""

    def __init__(
        self,
        lhs: ndarray | spmatrix | LinearOperator,
        rhs: ndarray | None = None,
        solution: ndarray | None = None,
        is_symmetric=False,
        is_positive_def=False,
        is_converged=None,
    ) -> None:
        """
        Args:
            lhs: The left-hand side (matrix or linear operator) of the problem.
            rhs: The right-hand side (vector) of the problem.
            solution: The current solution.
            is_symmetric: Whether to assume that the LHS is symmetric.
            is_positive_def: Whether to assume that the LHS is positive definite.
            is_converged: Whether the solution is converged to the specified tolerance.
                If False, the algorithm stopped before convergence.
                If None, no run was performed.
        """  # noqa: D205, D212, D415
        self.rhs = rhs
        self.lhs = lhs
        self.solution = solution
        self.is_converged = is_converged
        self.convergence_info = None
        self.is_symmetric = is_symmetric
        self.is_positive_def = is_positive_def

        if isinstance(lhs, LinearOperator):
            self.is_lhs_linear_operator = True
        else:
            self.is_lhs_linear_operator = False

        self.solver_options = None
        self.solver_name = None
        self.residuals_history = None

    def compute_residuals(
        self,
        relative_residuals=True,
        store=False,
        current_x=None,
    ) -> ndarray:
        """Compute the L2 norm of the residuals of the problem.

        Args:
            relative_residuals: If True, return norm(lhs.solution-rhs)/norm(rhs),
                else return norm(lhs.solution-rhs).
            store: Whether to store the residuals value in the residuals_history attribute.
            current_x: Compute the residuals associated with current_x,
                If None, compute then from the solution attribute.

        Returns:
            The residuals value.

        Raises:
            ValueError: If self.solution is None and current_x is None.
        """
        if self.rhs is None:
            raise ValueError("Missing RHS.")

        if current_x is None:
            current_x = self.solution
            if self.solution is None:
                raise ValueError("Missing solution.")

        res = norm(self.lhs.dot(current_x) - self.rhs)

        if relative_residuals:
            res /= norm(self.rhs)

        if store:
            if self.residuals_history is None:
                self.residuals_history = []
            self.residuals_history.append(res)

        return res

    def plot_residuals(self) -> Figure:
        """Plot the residuals' convergence in log scale.

        Returns:
            The matplotlib figure.

        Raises:
            ValueError: When the residuals' history is empty.
        """
        if self.residuals_history is None or len(self.residuals_history) == 0:
            raise ValueError(
                "Residuals history is empty. "
                " Use the 'store_residuals' option for the solver."
            )

        fig = plt.figure(figsize=(11.0, 6.0))
        plt.plot(self.residuals_history, color="black", lw=2)
        ax1 = fig.gca()
        ax1.set_yscale("log")
        ax1.set_title(f"Linear solver '{self.solver_name}' convergence")
        ax1.set_ylabel("Residuals norm (log)")
        ax1.set_xlabel("Iterations")
        return fig

    def check(self) -> None:
        """Check the consistency of the dimensions of the LHS and RHS.

        Raises:
            ValueError: When the shapes are inconsistent.
        """
        lhs_shape = self.lhs.shape
        rhs_shape = self.rhs.shape

        if (
            (len(lhs_shape) != 2)
            or (lhs_shape[0] != rhs_shape[0])
            or (len(rhs_shape) != 1 and rhs_shape[-1] != 1)
        ):
            raise ValueError(
                "Incompatible dimensions in linear system Ax=b,"
                " A shape is %s and b shape is %s",
                self.lhs.shape,
                self.rhs.shape,
            )
