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
#        :author: Benoit Pauwels
"""SciPy linear programming library wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Final

from numpy import concatenate
from numpy import isfinite
from scipy.optimize import linprog

from gemseo.algos.design_space_utils import get_value_and_bounds
from gemseo.algos.opt._base_optimization_library_settings import (
    BaseOptimizationLibrarySettings,
)
from gemseo.algos.opt.base_optimization_library import BaseOptimizationLibrary
from gemseo.algos.opt.base_optimization_library import OptimizationAlgorithmDescription
from gemseo.algos.opt.core.linear_constraints import build_constraints_matrices
from gemseo.algos.opt.scipy_linprog._base_scipy_linprog_settings import (
    BaseSciPyLinProgSettings,
)
from gemseo.algos.opt.scipy_linprog._settings.highs_dual_simplex import (
    HiGHSDualSimplexSettings,
)
from gemseo.algos.opt.scipy_linprog._settings.highs_interior_point import (
    HiGHSInteriorPointSettings,
)
from gemseo.algos.optimization_result import OptimizationResult
from gemseo.core.mdo_functions.mdo_linear_function import MDOLinearFunction
from gemseo.utils.compatibility.scipy import get_row
from gemseo.utils.compatibility.scipy import sparse_classes

if TYPE_CHECKING:
    from gemseo.algos.optimization_problem import OptimizationProblem


@dataclass
class ScipyLinProgAlgorithmDescription(OptimizationAlgorithmDescription):
    """The description of the SciPy linear programming library."""

    library_name: str = "SciPy Linear Programming"
    """The library name."""

    handle_equality_constraints: bool = True
    """Whether the optimization algorithm handles equality constraints."""

    handle_inequality_constraints: bool = True
    """Whether the optimization algorithm handles inequality constraints."""

    for_linear_problems: bool = True
    """Whether the optimization algorithm is dedicated to linear problems."""

    settings: type[BaseSciPyLinProgSettings] = BaseSciPyLinProgSettings
    """The option validation model for SciPy linear programming library."""


class ScipyLinprog(BaseOptimizationLibrary):
    """SciPy linear programming library interface.

    See BaseOptimizationLibrary.
    """

    _SUPPORT_SPARSE_JACOBIAN: ClassVar[bool] = True
    """Whether the library supports sparse Jacobians."""

    __DOC: Final[str] = "https://docs.scipy.org/doc/scipy/reference/"

    # TODO: Remove legacy methods "interior-point", "revised simplex" and "simplex".
    ALGORITHM_INFOS: ClassVar[dict[str, ScipyLinProgAlgorithmDescription]] = {
        "LINEAR_INTERIOR_POINT": ScipyLinProgAlgorithmDescription(
            algorithm_name="Linear interior point",
            description=(
                "Linear programming by the interior-point"
                " method implemented in the SciPy library"
            ),
            internal_algorithm_name="interior-point",
            website=f"{__DOC}optimize.linprog-interior-point.html",
        ),
        "REVISED_SIMPLEX": ScipyLinProgAlgorithmDescription(
            algorithm_name="Revised simplex",
            description=(
                "Linear programming by a two-phase revised"
                " simplex algorithm implemented in the SciPy library"
            ),
            internal_algorithm_name="revised simplex",
            website=f"{__DOC}optimize.linprog-revised_simplex.html",
        ),
        "SIMPLEX": ScipyLinProgAlgorithmDescription(
            algorithm_name="Simplex",
            description=(
                "Linear programming by the two-phase simplex"
                " algorithm implemented in the SciPy library"
            ),
            internal_algorithm_name="simplex",
            website=f"{__DOC}optimize.linprog-simplex.html",
        ),
        "HIGHS_INTERIOR_POINT": ScipyLinProgAlgorithmDescription(
            algorithm_name="Interior point method",
            description=("Linear programming using the HiGHS interior point solver."),
            internal_algorithm_name="highs-ipm",
            website=f"{__DOC}optimize.linprog-highs-ipm.html",
            settings=HiGHSInteriorPointSettings,
        ),
        "HIGHS_DUAL_SIMPLEX": ScipyLinProgAlgorithmDescription(
            algorithm_name="Dual simplex",
            description=("Linear programming using the HiGHS dual simplex solver."),
            internal_algorithm_name="highs-ds",
            website=f"{__DOC}optimize.linprog-highs-ds.html",
            settings=HiGHSDualSimplexSettings,
        ),
        "HIGHS": ScipyLinProgAlgorithmDescription(
            algorithm_name="HiGHS",
            description=(
                "Linear programming using the HiGHS solvers. "
                "A choice is automatically made between the dual simplex "
                "and the interior-point method."
            ),
            internal_algorithm_name="highs",
            website=f"{__DOC}optimize.linprog-highs.html",
        ),
    }

    def _run(self, problem: OptimizationProblem, **settings: Any) -> OptimizationResult:
        # Get the starting point and bounds
        x_0, l_b, u_b = get_value_and_bounds(problem.design_space, False)
        # Replace infinite bounds with None
        l_b = [val if isfinite(val) else None for val in l_b]
        u_b = [val if isfinite(val) else None for val in u_b]
        bounds = list(zip(l_b, u_b))

        # Build the functions matrices
        # N.B. use the non-processed functions to access the coefficients
        coefficients = problem.objective.original.coefficients
        if isinstance(coefficients, sparse_classes):
            obj_coeff = get_row(coefficients, 0).todense().flatten()
        else:
            obj_coeff = coefficients[0, :]

        ineq_lhs, ineq_rhs = build_constraints_matrices(
            problem.constraints.get_originals(),
            MDOLinearFunction.ConstraintType.INEQ,
        )
        eq_lhs, eq_rhs = build_constraints_matrices(
            problem.constraints.get_originals(),
            MDOLinearFunction.ConstraintType.EQ,
        )

        # Filter settings to get only the scipy.optimize.linprog ones
        settings_ = self._filter_settings(settings, BaseOptimizationLibrarySettings)

        linprog_result = linprog(
            c=obj_coeff.real,
            A_ub=ineq_lhs,
            b_ub=ineq_rhs,
            A_eq=eq_lhs,
            b_eq=eq_rhs,
            bounds=bounds,
            method=self.ALGORITHM_INFOS[self._algo_name].internal_algorithm_name,
            options=settings_,
            integrality=concatenate([
                [
                    problem.design_space.variable_types[variable_name]
                    == problem.design_space.DesignVariableType.INTEGER
                ]
                * problem.design_space.variable_sizes[variable_name]
                for variable_name in problem.design_space.variable_names
            ]),
        )

        # N.B. SciPy tolerance on bounds is higher than the DesignSpace one
        x_opt = problem.design_space.project_into_bounds(linprog_result.x)
        output_functions, jacobian_functions = problem.get_functions(
            jacobian_names=(),
            evaluate_objective=True,
            no_db_no_norm=True,
        )
        val_opt, jac_opt = problem.evaluate_functions(
            design_vector=x_opt,
            design_vector_is_normalized=False,
            output_functions=output_functions,
            jacobian_functions=jacobian_functions,
        )
        f_opt = val_opt[problem.objective.name]
        constraint_values = {
            key: val_opt[key] for key in problem.constraints.get_names()
        }
        constraints_grad = {
            key: jac_opt[key] for key in problem.constraints.get_names()
        }
        is_feasible = problem.constraints.is_point_feasible(val_opt)
        return OptimizationResult(
            x_0=x_0,
            x_0_as_dict=problem.design_space.convert_array_to_dict(x_0),
            x_opt=x_opt,
            x_opt_as_dict=problem.design_space.convert_array_to_dict(x_opt),
            f_opt=f_opt,
            objective_name=problem.objective.name,
            status=linprog_result.status,
            constraint_values=constraint_values,
            constraints_grad=constraints_grad,
            optimizer_name=self._algo_name,
            message=linprog_result.message,
            n_obj_call=None,
            n_grad_call=None,
            n_constr_call=None,
            is_feasible=is_feasible,
        )
