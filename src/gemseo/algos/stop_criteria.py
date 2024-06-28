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
"""Various termination criteria for drivers."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Any
from typing import Final

from numpy import all
from numpy import allclose
from numpy import average
from numpy import bool_
from numpy import ndarray
from numpy.linalg import norm

from gemseo.algos.lagrange_multipliers import LagrangeMultipliers

if TYPE_CHECKING:
    from gemseo.algos.optimization_problem import OptimizationProblem


class TerminationCriterion(Exception):  # noqa: N818
    """Stop driver for some reason."""


class FunctionIsNan(TerminationCriterion):  # noqa: N818
    """Stops driver when a function has NaN value or NaN Jacobian."""


class DesvarIsNan(TerminationCriterion):  # noqa: N818
    """Stops driver when the design variables are nan."""


class MaxIterReachedException(TerminationCriterion):  # noqa: N818
    """Exception raised when the maximum number of iterations is reached."""


class MaxTimeReached(TerminationCriterion):  # noqa: N818
    """Exception raised when the maximum execution time is reached."""


class FtolReached(TerminationCriterion):  # noqa: N818
    """Exception raised when the f_tol_rel or f_tol_abs criteria is reached."""


class XtolReached(TerminationCriterion):  # noqa: N818
    """Exception raised when the x_tol_rel or x_tol_abs criteria is reached."""


class KKTReached(TerminationCriterion):
    """A termination criterion based on the Karush-Kuhn-Tucker (KKT) residual norm."""


KKT_RESIDUAL_NORM: Final[str] = "KKT residual norm"
"""The name to store the KKT residual norm in a database."""


@dataclass
class BaseToleranceTester:
    """The base class to test the tolerance with respect to a reference value.

    The reference value corresponds to the coordinate-wise average of the values
    associated to the last iterations.
    """

    absolute: float = 0.0
    """The absolute tolerance."""

    relative: float = 0.0
    """The relative tolerance."""

    n_last_iterations: int = 3
    """The number of last points to compute the reference."""

    termination_criterion: TerminationCriterion = field(init=False)
    """The termination criterion."""

    def check(
        self, problem: OptimizationProblem, raise_exception: bool = False, **kwargs: Any
    ) -> bool:
        """Check whether the tolerance criterion is met.

        Args:
            problem: The optimization problem to which the database is attached.
            raise_exception: Whether to raise an exception
                when the tolerance criterion is not met.
            **kwargs: The options of the tester.

        Returns:
            Whether the tolerance criterion is not met.

        Raises:
            TerminationCriterion: When the tolerance criterion is not met
                and ``raise_exception`` is ``True``.
        """
        tolerance_criterion_is_reached = self._check(problem, **kwargs)
        if raise_exception and tolerance_criterion_is_reached:
            raise self.termination_criterion

        return tolerance_criterion_is_reached

    @abstractmethod
    def _check(self, problem: OptimizationProblem, *args: Any, **kwargs: Any) -> bool:
        """Check whether the tolerance criterion is met.

        Args:
            problem: The optimization problem to which the database is attached.
            **kwargs: The options of the tester.

        Returns:
            Whether the tolerance criterion is not met.
        """


@dataclass
class ObjectiveToleranceTester(BaseToleranceTester):
    """A tolerance tester for the objective."""

    termination_criterion: TerminationCriterion = field(default=FtolReached, init=False)

    def _check(self, problem: OptimizationProblem, *args: Any, **kwargs: Any) -> bool:  # noqa: D102
        database = problem.database
        if len(database) < self.n_last_iterations:
            return False

        # Checks that there is at least one feasible point
        x_values = database.get_last_n_x_vect(self.n_last_iterations)
        if not any(
            problem.constraints.is_point_feasible(database[x_val]) for x_val in x_values
        ):
            return False

        obj_name = problem.objective.name
        f_values = [
            f_value
            for f_value in [
                database.get_function_value(obj_name, x_val) for x_val in x_values
            ]
            if f_value is not None
        ]
        if len(f_values) < self.n_last_iterations:
            return False

        f_average = average(f_values)
        return all([
            allclose(f_val, f_average, atol=self.absolute, rtol=self.relative)
            for f_val in f_values
        ])


@dataclass
class DesignToleranceTester(BaseToleranceTester):
    """A tolerance tester for the design_vector."""

    termination_criterion: TerminationCriterion = field(default=XtolReached, init=False)

    def _check(self, problem: OptimizationProblem, *args: Any, **kwargs: Any) -> bool:  # noqa: D102
        database = problem.database
        if len(database) < self.n_last_iterations:
            return False

        x_values = database.get_last_n_x_vect(self.n_last_iterations)

        # Checks that there is at least one feasible point
        if not any(
            problem.constraints.is_point_feasible(database[x_val]) for x_val in x_values
        ):
            return False

        x_average = average(x_values, axis=0)
        return all([
            allclose(x_val, x_average, atol=self.absolute, rtol=self.relative)
            for x_val in x_values
        ])


@dataclass
class KKTConditionsTester(BaseToleranceTester):
    """A tester for the Karush-Kuhn-Tucker (KKT) conditions."""

    termination_criterion: TerminationCriterion = field(default=KKTReached, init=False)

    ineq_tolerance: float = 0.0
    """The tolerance for the inequality constraints."""

    kkt_norm: float = 0.0
    """The reference KKT norm."""

    def _check(self, problem: OptimizationProblem, input_vector: ndarray) -> bool:  # noqa: D102
        return kkt_residual_computation(
            problem, input_vector, self.ineq_tolerance
        ) <= max(self.absolute, self.relative * self.kkt_norm)


def kkt_residual_computation(
    opt_problem: OptimizationProblem,
    x_vect: ndarray,
    ineq_tolerance: float = 1e-4,
) -> float:
    """Compute the KKT residual norm.

    This implementation is inspired from Svanberg Matlab implementation of
    MMA algorithm see :cite:`svanberg1998method`

    Args:
        opt_problem: The optimization problem containing an optimization history.
        x_vect: The design point vector where the KKT conditions are tested.
        ineq_tolerance: The tolerance to consider a constraint as active.

    Returns:
        The KKT residual norm.
    """
    res = opt_problem.database.get_function_value(KKT_RESIDUAL_NORM, x_vect)
    if res is not None:
        return res
    lagrange = LagrangeMultipliers(opt_problem)
    if opt_problem.constraints:
        lagrange.compute(x_vect, ineq_tolerance=ineq_tolerance)
        res = lagrange.kkt_residual + lagrange.constraint_violation
        opt_problem.database.store(x_vect, {KKT_RESIDUAL_NORM: res})
        return res

    res = norm(lagrange.get_objective_jacobian(x_vect))
    opt_problem.database.store(x_vect, {KKT_RESIDUAL_NORM: res})
    return res


def is_x_tol_reached(
    opt_problem: OptimizationProblem,
    x_tol_rel: float = 1e-6,
    x_tol_abs: float = 1e-6,
    n_x: int = 2,
) -> bool | bool_:
    """Tests if the tolerance on the design variables are reached.

    The coordinate wise
    average of the last n_x points are taken Then it is checked that all points are
    within the distance of the center with relative and absolute tolerances specified by
    the user.

    Parameters
    ----------
    opt_problem: OptimizationProblem
        the optimization problem containing the iterations
    x_tol_rel: float
        relative tolerance
    x_tol_abs: float
        absolute tolerance
    n_x: int
        number of design vectors to account for
    """
    tester = DesignToleranceTester(
        absolute=x_tol_abs, relative=x_tol_rel, n_last_iterations=n_x
    )
    return tester.check(opt_problem)


def is_f_tol_reached(
    opt_problem: OptimizationProblem,
    f_tol_rel: float = 1e-6,
    f_tol_abs: float = 1e-6,
    n_x: int = 2,
) -> bool | bool_:
    """Tests if the tolerance on the objective function are reached.

    The average function
    value of the last n_x points are taken Then it is checked that all points are within
    the distance of the center with relative and absolute tolerances specified by the
    user.

    Parameters
    ----------
    opt_problem: OptimizationProblem
        the optimization problem containing the iterations
    f_tol_rel: float
        relative tolerance
    f_tol_abs: float
        absolute tolerance
    n_x: int
        number of design vectors to account for
    """
    tester = ObjectiveToleranceTester(
        absolute=f_tol_abs, relative=f_tol_rel, n_last_iterations=n_x
    )
    return tester.check(opt_problem)


def is_kkt_residual_norm_reached(
    opt_problem: OptimizationProblem,
    x_vect: ndarray,
    kkt_abs_tol: float = 0.0,
    kkt_rel_tol: float = 0.0,
    ineq_tolerance: float = 1e-4,
    reference_residual: float = 1.0,
) -> bool:
    """Test if the KKT conditions are satisfied.

    Args:
        opt_problem: The optimization problem containing an optimization history.
        x_vect: The design point vector where the KKT conditions are tested.
        kkt_abs_tol: The absolute tolerance on the KKT condition residual.
        kkt_rel_tol: The relative tolerance on the KKT condition residual.
        ineq_tolerance: The tolerance to consider a constraint as active.
        reference_residual: The reference KKT condition residual.

    Returns:
        Whether the absolute or the relative KKT residual norm criterion is reached.
    """
    tester = KKTConditionsTester(
        absolute=kkt_abs_tol,
        relative=kkt_rel_tol,
        ineq_tolerance=ineq_tolerance,
        kkt_norm=reference_residual,
    )
    return tester.check(opt_problem, input_vector=x_vect)
