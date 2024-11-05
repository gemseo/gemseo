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
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""The library of SciPy global optimization algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from sys import maxsize
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Final
from typing import Union

from numpy import float64
from numpy import inf as np_inf
from numpy import int32
from numpy import isfinite
from numpy import real
from numpy.typing import NDArray
from scipy.optimize import NonlinearConstraint
from scipy.optimize import differential_evolution
from scipy.optimize import dual_annealing
from scipy.optimize import shgo

from gemseo.algos.design_space_utils import get_value_and_bounds
from gemseo.algos.opt.base_optimization_library import BaseOptimizationLibrary
from gemseo.algos.opt.base_optimization_library import OptimizationAlgorithmDescription
from gemseo.algos.opt.base_optimizer_settings import BaseOptimizerSettings
from gemseo.algos.opt.scipy_global.settings.base_scipy_global_settings import (
    BaseSciPyGlobalSettings,
)
from gemseo.algos.opt.scipy_global.settings.differential_evolution import (
    DIFFERENTIAL_EVOLUTION_Settings,
)
from gemseo.algos.opt.scipy_global.settings.dual_annealing import (
    DUAL_ANNEALING_Settings,
)
from gemseo.algos.opt.scipy_global.settings.shgo import SHGO_Settings

if TYPE_CHECKING:
    from collections.abc import Callable

    from gemseo.algos.optimization_problem import OptimizationProblem
    from gemseo.core.mdo_functions.mdo_function import OutputType
    from gemseo.core.mdo_functions.mdo_function import WrappedFunctionType
    from gemseo.core.mdo_functions.mdo_function import WrappedJacobianType

InputType = NDArray[Union[float64, int32]]


@dataclass
class SciPyGlobalAlgorithmDescription(OptimizationAlgorithmDescription):
    """The description of a global optimization algorithm from the SciPy library."""

    library_name: str = "SciPy Global Optimization"
    """The library name."""

    handle_integer_variables: bool = True
    """Whether the optimization algorithm handles integer variables."""

    Settings: type[BaseSciPyGlobalSettings] = BaseSciPyGlobalSettings
    """The option validation model for SciPy global optimization library."""


class ScipyGlobalOpt(BaseOptimizationLibrary):
    """The library of SciPy global optimization algorithms."""

    __NAMES_TO_FUNCTIONS: ClassVar[dict[str, Callable]] = {
        "DIFFERENTIAL_EVOLUTION": differential_evolution,
        "DUAL_ANNEALING": dual_annealing,
        "SHGO": shgo,
    }
    """The mapping between the algorithm name and the SciPy function."""

    __DOC: Final[str] = "https://docs.scipy.org/doc/scipy/reference/generated/"

    ALGORITHM_INFOS: ClassVar[dict[str, SciPyGlobalAlgorithmDescription]] = {
        "DUAL_ANNEALING": SciPyGlobalAlgorithmDescription(
            algorithm_name="Dual annealing",
            description="Dual annealing",
            internal_algorithm_name="dual_annealing",
            website=f"{__DOC}scipy.optimize.dual_annealing.html",
            Settings=DUAL_ANNEALING_Settings,
        ),
        "SHGO": SciPyGlobalAlgorithmDescription(
            algorithm_name="SHGO",
            description="Simplicial homology global optimization",
            handle_equality_constraints=True,
            handle_inequality_constraints=True,
            internal_algorithm_name="shgo",
            positive_constraints=True,
            website=f"{__DOC}scipy.optimize.shgo.html",
            Settings=SHGO_Settings,
        ),
        "DIFFERENTIAL_EVOLUTION": SciPyGlobalAlgorithmDescription(
            algorithm_name="Differential evolution",
            description="Differential Evolution algorithm",
            handle_equality_constraints=True,
            handle_inequality_constraints=True,
            internal_algorithm_name="differential_evolution",
            website=f"{__DOC}scipy.optimize.differential_evolution.html",
            Settings=DIFFERENTIAL_EVOLUTION_Settings,
        ),
    }

    def _iter_callback(self, x_vect: InputType) -> None:
        """Call the objective and constraints functions.

        Args:
            x_vect: The input data with which to call the functions.
        """
        if self._normalize_ds:
            x_vect = self._problem.design_space.normalize_vect(x_vect)

        self._problem.objective.evaluate(x_vect)
        for constraint in self._problem.constraints:
            constraint.evaluate(x_vect)

    def _compute_objective(self, x_vect: InputType) -> OutputType:
        """Wrap the objective function of the problem to pass it to SciPy.

        Cast the result to real.

        Args:
            x_vect: The input data with which to call the function.
        """
        return real(self._problem.objective.evaluate(x_vect))

    def _run(self, problem: OptimizationProblem, **settings: Any) -> tuple[str, Any]:
        # Get the normalized bounds:
        _, l_b, u_b = get_value_and_bounds(problem.design_space, self._normalize_ds)
        # Replace infinite values with None:
        l_b = [val if isfinite(val) else None for val in l_b]
        u_b = [val if isfinite(val) else None for val in u_b]
        bounds = list(zip(l_b, u_b))

        # This is required because some algorithms do not
        # call the objective very often when the problem
        # is very constrained (Power2) and OptProblem may fail
        # to detect the optimum.
        if problem.constraints:
            problem.add_listener(self._iter_callback)

        # Filter settings to get only the ones of the global optimizer
        settings_ = self._filter_settings(settings, BaseOptimizerSettings)

        if self._algo_name == "SHGO":
            constraints = self.__get_constraints_as_scipy_dictionary(problem)
            settings_["constraints"] = constraints
        elif self._algo_name == "DIFFERENTIAL_EVOLUTION":
            constraints = self.__get_non_linear_constraints(problem)
            settings_["constraints"] = constraints

        # Deactivate stopping criteria which are handled by GEMSEO
        if self._algo_name == "SHGO":
            settings_["options"].update(
                dict.fromkeys(["maxev", "maxfev", "maxiter", "maxtime"], maxsize)
            )
            settings_["options"]["ftol"] = 0.0
        elif self._algo_name == "DUAL_ANNEALING":
            settings_["maxiter"] = settings_["maxfun"] = maxsize
        else:  # Necessarily the differential evolution algorithm
            settings_["maxiter"] = maxsize

        global_optimizer = self.__NAMES_TO_FUNCTIONS[self._algo_name]
        opt_result = global_optimizer(
            func=self._compute_objective,
            bounds=bounds,
            **settings_,
        )

        return opt_result.message, opt_result.success

    @staticmethod
    def __get_non_linear_constraints(
        problem: OptimizationProblem,
    ) -> tuple[NonlinearConstraint, ...]:
        """Return the SciPy nonlinear constraints.

        Args:
            problem: The problem to be solved.

        Returns:
            The SciPy nonlinear constraints.
        """
        eq_tolerance = problem.tolerances.equality
        constraints = [
            NonlinearConstraint(
                constr.evaluate, -eq_tolerance, eq_tolerance, jac=constr.jac
            )
            for constr in problem.constraints.get_equality_constraints()
        ]
        ineq_tolerance = problem.tolerances.inequality
        constraints.extend([
            NonlinearConstraint(
                constr.evaluate, -np_inf, ineq_tolerance, jac=constr.jac
            )
            for constr in problem.constraints.get_inequality_constraints()
        ])
        return tuple(constraints)

    def __get_constraints_as_scipy_dictionary(
        self, problem: OptimizationProblem
    ) -> list[dict[str, str | WrappedFunctionType | WrappedJacobianType]]:
        """Create the constraints to be passed to a SciPy algorithm as dictionaries.

        Args:
            problem: The problem to be solved.

        Returns:
            The constraints.
        """
        return [
            {
                "type": constraint.f_type,
                "fun": constraint.evaluate,
                "jac": constraint.jac,
            }
            for constraint in self._get_right_sign_constraints(problem)
        ]
