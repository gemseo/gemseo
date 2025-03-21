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
#                           documentation
#        :author: Damien Guenot
#        :author: Francois Gallard, refactoring
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Base class for libraries of optimizers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Final

import numpy
from numpy import isinf

from gemseo.algos._unsuitability_reason import _UnsuitabilityReason
from gemseo.algos.base_driver_library import BaseDriverLibrary
from gemseo.algos.base_driver_library import DriverDescription
from gemseo.algos.opt.base_optimizer_settings import BaseOptimizerSettings
from gemseo.algos.stop_criteria import DesignToleranceTester
from gemseo.algos.stop_criteria import KKTConditionsTester
from gemseo.algos.stop_criteria import ObjectiveToleranceTester
from gemseo.algos.stop_criteria import kkt_residual_computation

if TYPE_CHECKING:
    from numpy import ndarray

    from gemseo.algos.optimization_problem import OptimizationProblem
    from gemseo.core.mdo_functions.mdo_function import MDOFunction


@dataclass
class OptimizationAlgorithmDescription(DriverDescription):
    """The description of an optimization algorithm."""

    handle_equality_constraints: bool = False
    """Whether the optimization algorithm handles equality constraints."""

    handle_inequality_constraints: bool = False
    """Whether the optimization algorithm handles inequality constraints."""

    handle_multiobjective: bool = False
    """Whether the optimization algorithm handles multiple objectives."""

    positive_constraints: bool = False
    """Whether the optimization algorithm requires positive constraints."""

    for_linear_problems: bool = False
    """Whether the optimization algorithm is dedicated to linear problems."""

    require_gradient: bool = False
    """Whether the optimization algorithm requires the gradient."""

    Settings: type[BaseOptimizerSettings] = BaseOptimizerSettings
    """The settings validation model."""


class BaseOptimizationLibrary(BaseDriverLibrary):
    """Base class for libraries of optimizers.

    Typically used as:

    #. Instantiate an :class:`.BaseOptimizationLibrary`.
    #. Select the algorithm with :attr:`._algo_name`.
    #. Solve an :class:`.OptimizationProblem` with :meth:`.execute`.

    Notes:
        The missing current values
        of the :class:`.DesignSpace` attached to the :class:`.OptimizationProblem`
        are automatically initialized
        with the method :meth:`.DesignSpace.initialize_missing_current_values`.
    """

    # Option names
    _F_TOL_REL: Final[str] = "ftol_rel"
    _F_TOL_ABS: Final[str] = "ftol_abs"
    _KKT_TOL_ABS: Final[str] = "kkt_tol_abs"
    _KKT_TOL_REL: Final[str] = "kkt_tol_rel"
    _MAX_ITER: Final[str] = "max_iter"
    _SCALING_THRESHOLD: Final[str] = "scaling_threshold"
    _STOP_CRIT_NX: Final[str] = "stop_crit_n_x"
    _X_TOL_REL: Final[str] = "xtol_rel"
    _X_TOL_ABS: Final[str] = "xtol_abs"

    _f_tol_tester: ObjectiveToleranceTester
    """A tester for the termination criterion associated to the objective function."""

    _x_tol_tester: DesignToleranceTester
    """A tester for the termination criterion associated to the design variables."""

    ALGORITHM_INFOS: ClassVar[dict[str, OptimizationAlgorithmDescription]] = {}
    """The description of the algorithms contained in the library."""

    def __init__(self, algo_name: str) -> None:  # noqa:D107
        super().__init__(algo_name)
        self._f_tol_tester = ObjectiveToleranceTester()
        self._x_tol_tester = DesignToleranceTester()

    def _check_constraints_handling(self, problem: OptimizationProblem) -> None:
        """Check if problem and algorithm are consistent for constraints handling."""
        algo_name = self._algo_name
        if (
            tuple(problem.constraints.get_equality_constraints())
            and not self.ALGORITHM_INFOS[algo_name].handle_equality_constraints
        ):
            msg = (
                "Requested optimization algorithm "
                f"{algo_name} can not handle equality constraints."
            )
            raise ValueError(msg)
        if (
            tuple(problem.constraints.get_inequality_constraints())
            and not self.ALGORITHM_INFOS[algo_name].handle_inequality_constraints
        ):
            msg = (
                "Requested optimization algorithm "
                f"{algo_name} can not handle inequality constraints."
            )
            raise ValueError(msg)

    def _get_right_sign_constraints(self, problem: OptimizationProblem):
        """Transform the problem constraints into their opposite sign counterpart.

        This is done if the algorithm requires positive constraints.

        Args:
            problem: The problem to be solved.

        Returns:
            The constraints with the right sign.
        """
        if (
            tuple(problem.constraints.get_inequality_constraints())
            and self.ALGORITHM_INFOS[self._algo_name].positive_constraints
        ):
            return [-constraint for constraint in problem.constraints]
        return problem.constraints

    def _pre_run(self, problem: OptimizationProblem, **settings: Any) -> None:
        super()._pre_run(problem, **settings)

        self._check_constraints_handling(problem)

        n_points = settings[self._STOP_CRIT_NX]

        self._f_tol_tester = ObjectiveToleranceTester(
            absolute=settings[self._F_TOL_ABS],
            relative=settings[self._F_TOL_REL],
            n_last_iterations=n_points,
        )

        self._x_tol_tester = DesignToleranceTester(
            absolute=settings[self._X_TOL_ABS],
            relative=settings[self._X_TOL_REL],
            n_last_iterations=n_points,
        )

        self._init_iter_observer(problem, settings[self._MAX_ITER])

        require_gradient = self.ALGORITHM_INFOS[self._algo_name].require_gradient
        if require_gradient:
            kkt_abs_tol = settings[self._KKT_TOL_ABS]
            kkt_rel_tol = settings[self._KKT_TOL_REL]
            if not isinf(kkt_abs_tol) or not isinf(kkt_rel_tol):
                problem.add_listener(
                    _KKTChecker(
                        problem,
                        kkt_abs_tol,
                        kkt_rel_tol,
                        settings[self._INEQ_TOLERANCE],
                    ),
                    at_each_iteration=False,
                    at_each_function_call=True,
                )

        problem.design_space.initialize_missing_current_values()
        if problem.differentiation_method == self.DifferentiationMethod.COMPLEX_STEP:
            problem.design_space.to_complex()

        # First, evaluate all functions at x_0. Some algorithms don't do this
        output_functions, jacobian_functions = problem.get_functions(
            jacobian_names=() if require_gradient else None,
            observable_names=None,
        )

        function_values, _ = problem.evaluate_functions(
            design_vector_is_normalized=self._normalize_ds,
            output_functions=output_functions or None,
            jacobian_functions=jacobian_functions or None,
        )

        scaling_threshold = settings[self._SCALING_THRESHOLD]
        if scaling_threshold is not None:
            self._problem.objective = self.__scale(
                self._problem.objective,
                function_values[self._problem.objective.name],
                scaling_threshold,
            )
            self._problem.constraints = [
                self.__scale(
                    constraint, function_values[constraint.name], scaling_threshold
                )
                for constraint in self._problem.constraints
            ]

    @classmethod
    def _get_unsuitability_reason(
        cls,
        algorithm_description: OptimizationAlgorithmDescription,
        problem: OptimizationProblem,
    ) -> _UnsuitabilityReason:
        reason = super()._get_unsuitability_reason(algorithm_description, problem)
        if reason:
            return reason

        if (
            tuple(problem.constraints.get_equality_constraints())
            and not algorithm_description.handle_equality_constraints
        ):
            return _UnsuitabilityReason.EQUALITY_CONSTRAINTS

        if (
            tuple(problem.constraints.get_inequality_constraints())
            and not algorithm_description.handle_inequality_constraints
        ):
            return _UnsuitabilityReason.INEQUALITY_CONSTRAINTS

        if not problem.is_linear and algorithm_description.for_linear_problems:
            return _UnsuitabilityReason.NON_LINEAR_PROBLEM

        return reason

    def _new_iteration_callback(self, x_vect: ndarray) -> None:
        super()._new_iteration_callback(x_vect)
        self._f_tol_tester.check(self._problem, raise_exception=True)
        self._x_tol_tester.check(self._problem, raise_exception=True)

    @staticmethod
    def __scale(
        function: MDOFunction,
        function_value: ndarray,
        scaling_threshold: float,
    ) -> MDOFunction:
        """Scale a function based on its value on the current design values.

        Args:
            function: The function.
            function_value: The function value of reference for scaling.
            scaling_threshold: The threshold on the reference function value
                that triggers scaling.

        Returns:
            The scaled function.
        """
        reference_values = numpy.absolute(function_value)
        threshold_reached = reference_values > scaling_threshold
        if not threshold_reached.any():
            return function

        scaled_function = function / numpy.where(
            threshold_reached, reference_values, 1.0
        )
        # Use same function name for consistency with name used in database
        scaled_function.name = function.name
        return scaled_function


class _KKTChecker:
    """A functor to verify the KKT norm stopping criterion."""

    def __init__(
        self,
        problem: OptimizationProblem,
        kkt_abs_tol: float,
        kkt_rel_tol: float,
        ineq_tolerance: float,
    ) -> None:
        """
        Args:
            problem: The optimization problem.
            kkt_abs_tol: The absolute tolerance for the KKT conditions.
            kkt_rel_tol: The relative tolerance for the KKT conditions.
            ineq_tolerance: The absolute tolerance for the inequality constraints.
        """  # noqa: D205, D212
        self.__problem = problem
        self.__kkt_tester = KKTConditionsTester(
            absolute=0.0 if isinf(kkt_abs_tol) else kkt_abs_tol,
            relative=0.0 if isinf(kkt_rel_tol) else kkt_rel_tol,
            ineq_tolerance=ineq_tolerance,
        )

    def __call__(self, input_value: ndarray) -> None:
        """Verify the KKT norm stopping criterion.

        Args:
            input_value: The input value.

        Raises:
            KKTReached: If the absolute tolerance on the KKT residual is reached.
        """
        check_kkt = True
        function_names = [
            self.__problem.standardized_objective_name,
            *self.__problem.constraints.get_names(),
        ]
        database = self.__problem.database
        for function_name in function_names:
            if (
                database.get_function_value(
                    database.get_gradient_name(function_name), input_value
                )
                is None
            ) or (database.get_function_value(function_name, input_value) is None):
                check_kkt = False
                break

        if check_kkt:
            if not self.__kkt_tester.kkt_norm:
                self.__kkt_tester.kkt_norm = kkt_residual_computation(
                    self.__problem, input_value, self.__kkt_tester.ineq_tolerance
                )

            self.__kkt_tester.check(
                self.__problem, raise_exception=True, input_vector=input_value
            )
