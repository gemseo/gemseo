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

from typing import TYPE_CHECKING
from typing import Any

from matplotlib.pyplot import figure
from matplotlib.pyplot import plot
from numpy.linalg import norm
from scipy.sparse.linalg import LinearOperator

from gemseo.algos.base_problem import BaseProblem

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from numpy import floating

    from gemseo.typing import NumberArray
    from gemseo.typing import SparseOrDenseRealArray


class LinearProblem(BaseProblem):
    r"""Representation of the linear equations' system :math:`Ax = b`.

    It also contains the solution, and some properties of the system such as the
    symmetry or positive definiteness.
    """

    rhs: NumberArray | None
    """The right-hand side of the equation."""

    lhs: LinearOperator | SparseOrDenseRealArray
    """The left-hand side of the equation.

    If ``None``, the problem can't be solved and the user has to set it after init.
    """

    solution: NumberArray | None
    """The current solution of the problem."""

    is_converged: bool | None
    """Whether the solution satisfies the specified tolerance.

    If ``None``, no run was performed.
    """

    convergence_info: int | str
    """The information provided by the solver if convergence occurred or not."""

    is_symmetric: bool
    """Whether the LHS is symmetric."""

    is_positive_def: bool
    """Whether the LHS is positive definite."""

    is_lhs_linear_operator: bool
    """Whether the LHS is a linear operator."""

    solver_name: str
    """The solver name."""

    residuals_history: list[floating[Any]]
    """The convergence history of residuals."""

    def __init__(
        self,
        lhs: SparseOrDenseRealArray | LinearOperator,
        rhs: NumberArray | None = None,
        solution: NumberArray | None = None,
        is_symmetric: bool = False,
        is_positive_def: bool = False,
        is_converged: bool | None = None,
    ) -> None:
        """
        Args:
            lhs: The left-hand side (matrix or linear operator) of the problem.
            rhs: The right-hand side (vector) of the problem.
            solution: The current solution.
            is_symmetric: Whether the left-hand side is symmetric.
            is_positive_def: Whether the left-hand side is positive definite.
            is_converged: Whether the solution is converged to the specified tolerance.
                If ``False``, the algorithm stopped before convergence.
                If ``None``, no run was performed.
        """  # noqa: D205, D212, D415
        self.rhs = rhs
        self.lhs = lhs
        self.solution = solution

        self.is_converged = is_converged
        self.convergence_info = ""

        self.is_symmetric = is_symmetric
        self.is_positive_def = is_positive_def
        self.is_lhs_linear_operator = isinstance(lhs, LinearOperator)

        self.solver_name = None
        self.residuals_history = []

    def compute_residuals(
        self,
        relative_residuals: bool = True,
        store: bool = False,
        current_x: NumberArray | None = None,
    ) -> floating[Any]:
        r"""Compute the Euclidean norm of the residual.

        Args:
            relative_residuals: If ``True``, one computes
                :math:` \|A x_k - b\|_2 /  \|b\|_2`, else :math:` \|A x_k - b\|_2`.
            store: Whether to store the residual norm in the history.
            current_x: Compute the residuals associated with current_x,
                If ``None``, compute then from the solution attribute.

        Returns:
            The residual norm.

        Raises:
            ValueError: If :attr:`.rhd` is ``None``.
            ValueError: If :attr:`.solution` is ``None`` and ``current_x`` is ``None``.
        """
        if self.rhs is None:
            msg = "No right-hand side available to compute residual."
            raise ValueError(msg)

        if current_x is None and self.solution is None:
            msg = "Neither solution or current iterate available to compute residual."
            raise ValueError(msg)

        x = self.solution if current_x is None else current_x
        residual_norm = norm(self.lhs.dot(x) - self.rhs)

        if relative_residuals:
            residual_norm /= norm(self.rhs)

        if store:
            self.residuals_history.append(residual_norm)

        return residual_norm

    def plot_residuals(self) -> Figure:
        """Plot the residuals' convergence in log scale.

        Returns:
            The matplotlib figure.

        Raises:
            ValueError: When the residual norm history is empty.
        """
        if len(self.residuals_history) == 0:
            msg = (
                "Residuals history is empty. "
                "Consider setting the 'store' attribute to `True`."
            )
            raise ValueError(msg)

        fig = figure(figsize=(11.0, 6.0))
        plot(self.residuals_history, color="black", lw=2)
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
            or (lhs_shape[1] != rhs_shape[0])
            or (len(rhs_shape) != 1 and rhs_shape[-1] != 1)
        ):
            msg = (
                "Incompatible dimensions in linear system Ax=b. "
                f"A shape is {lhs_shape} and b shape is {rhs_shape}"
            )
            raise ValueError(msg)
