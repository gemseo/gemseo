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
#        :author: Simone Coniglio
"""SciPy linear programming library wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from numpy import concatenate
from numpy import inf
from numpy import ones_like
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import milp

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.design_space_utils import get_value_and_bounds
from gemseo.algos.opt.base_optimization_library import BaseOptimizationLibrary
from gemseo.algos.opt.base_optimization_library import OptimizationAlgorithmDescription
from gemseo.algos.opt.base_optimizer_settings import BaseOptimizerSettings
from gemseo.algos.opt.core.linear_constraints import build_constraints_matrices
from gemseo.algos.opt.scipy_milp.settings.scipy_milp_settings import SciPyMILP_Settings
from gemseo.algos.optimization_result import OptimizationResult
from gemseo.core.mdo_functions.mdo_linear_function import MDOLinearFunction
from gemseo.utils.compatibility.scipy import get_row
from gemseo.utils.compatibility.scipy import sparse_classes

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gemseo.algos.optimization_problem import OptimizationProblem
    from gemseo.typing import RealArray


@dataclass
class ScipyMILPAlgorithmDescription(OptimizationAlgorithmDescription):
    """The description of a the SciPy mixed-integer linear programming library."""

    library_name: str = "SciPy Mixed-Integer Linear Programming"
    """The library name."""

    handle_equality_constraints: bool = True
    """Whether the optimization algorithm handles equality constraints."""

    handle_inequality_constraints: bool = True
    """Whether the optimization algorithm handles inequality constraints."""

    handle_integer_variables: bool = True
    """Whether the optimization algorithm handles integer variables."""

    Settings: type[SciPyMILP_Settings] = SciPyMILP_Settings
    """The option validation model for SciPy linear programming library."""


class ScipyMILP(BaseOptimizationLibrary):
    """SciPy Mixed Integer Linear Programming library interface.

    See BaseOptimizationLibrary.

    With respect to scipy milp function, this wrapper only allows continuous or integer
    variables.
    """

    ALGORITHM_INFOS: ClassVar[dict[str, ScipyMILPAlgorithmDescription]] = {
        "Scipy_MILP": ScipyMILPAlgorithmDescription(
            algorithm_name="Branch & Cut algorithm",
            description="Mixed-integer linear programming",
            internal_algorithm_name="milp",
            website="https://docs.scipy.org/doc/scipy/reference/generated/"
            "scipy.optimize.milp.html",
            Settings=SciPyMILP_Settings,
        ),
    }

    _SUPPORT_SPARSE_JACOBIAN: ClassVar[bool] = True
    """Whether the library support sparse Jacobians."""

    def __init__(self, algo_name: str = "Scipy_MILP") -> None:  # noqa:D107
        super().__init__(algo_name)

    def _run(
        self, problem: OptimizationProblem, **settings: Any
    ) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
        # Get the starting point and bounds
        x_0, l_b, u_b = get_value_and_bounds(problem.design_space, False)
        # Replace infinite bounds with None
        bounds = Bounds(lb=l_b, ub=u_b, keep_feasible=True)

        # Build the functions matrices
        # N.B. use the non-processed functions to access the coefficients
        coefficients = problem.objective.original.coefficients
        if isinstance(coefficients, sparse_classes):
            obj_coeff = get_row(coefficients, 0).todense().flatten()
        else:
            obj_coeff = coefficients[0, :]

        lq_constraints = []

        ineq_lhs, ineq_rhs = build_constraints_matrices(
            problem.constraints.get_originals(), MDOLinearFunction.ConstraintType.INEQ
        )
        if ineq_lhs is not None:
            lq_constraints.append(
                LinearConstraint(
                    ineq_lhs,
                    -inf * ones_like(ineq_rhs),
                    ineq_rhs + problem.tolerances.inequality,
                    keep_feasible=True,
                )
            )

        eq_lhs, eq_rhs = build_constraints_matrices(
            problem.constraints.get_originals(), MDOLinearFunction.ConstraintType.EQ
        )
        if eq_lhs is not None:
            lq_constraints.append(
                LinearConstraint(
                    eq_lhs,
                    eq_rhs - problem.tolerances.equality,
                    eq_rhs + problem.tolerances.equality,
                    keep_feasible=True,
                )
            )

        # Filter settings to get only the scipy.optimize.milp ones
        settings_ = self._filter_settings(settings, BaseOptimizerSettings)

        # Deactivate stopping criteria which are handled by GEMSEO
        settings["time_limit"] = inf

        # Pass the MILP to Scipy
        milp_result = milp(
            c=obj_coeff.real,
            bounds=bounds,
            constraints=lq_constraints,
            options=settings_,
            integrality=concatenate([
                [
                    self._problem.design_space.get_type(variable_name)
                    == DesignSpace.DesignVariableType.INTEGER
                ]
                * self._problem.design_space.get_size(variable_name)
                for variable_name in self._problem.design_space
            ]),
        )

        # Gather the optimization results
        x_opt = x_0 if milp_result.x is None else milp_result.x

        # N.B. SciPy tolerance on bounds is higher than the DesignSpace one
        x_opt = problem.design_space.project_into_bounds(x_opt)
        output_functions, jacobian_functions = problem.get_functions(
            jacobian_names=(),
            no_db_no_norm=True,
        )

        output_opt, jac_opt = problem.evaluate_functions(
            design_vector=x_opt,
            design_vector_is_normalized=False,
            output_functions=output_functions or None,
            jacobian_functions=jacobian_functions or None,
        )
        return None, None, output_opt, jac_opt, x_0, x_opt, milp_result

    def _get_result(
        self,
        problem: OptimizationProblem,
        message: Any,
        status: Any,
        output_opt: Mapping[str, RealArray],
        jac_opt: Mapping[str, RealArray],
        x_0: RealArray,
        x_opt: RealArray,
        result: Any,
    ) -> OptimizationResult:
        """
        Args:
            output_opt: The output values at optimum.
            jac_opt: The Jacobian values at optimum.
            x_0: The initial design value.
            x_opt: The optimal design value.
            result: A result specific to this library.
        """  # noqa: D205 D212
        f_opt = output_opt[problem.objective.name]
        constraint_names = list(problem.constraints.original_to_current_names.keys())
        constraint_values = {key: output_opt[key] for key in constraint_names}
        constraints_grad = {key: jac_opt[key] for key in constraint_names}
        is_feasible = problem.constraints.is_point_feasible(output_opt)
        return OptimizationResult(
            x_0=x_0,
            x_0_as_dict=problem.design_space.convert_array_to_dict(x_0),
            x_opt=x_opt,
            x_opt_as_dict=problem.design_space.convert_array_to_dict(x_opt),
            f_opt=f_opt,
            objective_name=problem.objective.name,
            status=result.status,
            constraint_values=constraint_values,
            constraints_grad=constraints_grad,
            optimizer_name=self._algo_name,
            message=result.message,
            n_obj_call=None,
            n_grad_call=None,
            n_constr_call=None,
            is_feasible=is_feasible,
        )
