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
from typing import Final

from numpy import concatenate
from numpy import inf
from numpy import ones_like
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import milp

from gemseo.algos.design_space_utils import get_value_and_bounds
from gemseo.algos.opt.base_optimization_library import BaseOptimizationLibrary
from gemseo.algos.opt.base_optimization_library import OptimizationAlgorithmDescription
from gemseo.algos.opt.core.linear_constraints import build_constraints_matrices
from gemseo.algos.optimization_result import OptimizationResult
from gemseo.core.mdofunctions.mdo_linear_function import MDOLinearFunction
from gemseo.utils.compatibility.scipy import get_row
from gemseo.utils.compatibility.scipy import sparse_classes

if TYPE_CHECKING:
    from gemseo.algos.optimization_problem import OptimizationProblem


@dataclass
class ScipyMILPAlgorithmDescription(OptimizationAlgorithmDescription):
    """The description of a MILP optimization algorithm from the SciPy library."""

    for_linear_problems: bool = False
    handle_equality_constraints: bool = True
    handle_inequality_constraints: bool = True
    library_name: str = "SciPy"
    handle_integer_variables: bool = True


class ScipyMILP(BaseOptimizationLibrary):
    """SciPy Mixed Integer Linear Programming library interface.

    See BaseOptimizationLibrary.

    With respect to scipy milp function, this wrapper only allows continuous or integer
    variables.
    """

    _OPTIONS_MAP: ClassVar[dict[int, str]] = {
        BaseOptimizationLibrary._MAX_ITER: "node_limit",
    }

    _SUPPORT_SPARSE_JACOBIAN: ClassVar[bool] = True
    """Whether the library support sparse Jacobians."""

    __DOC: Final[str] = "https://docs.scipy.org/doc/scipy/reference/"
    ALGORITHM_INFOS: ClassVar[dict[str, ScipyMILPAlgorithmDescription]] = {
        "Scipy_MILP": ScipyMILPAlgorithmDescription(
            algorithm_name="Branch & Cut algorithm",
            description=("Mixed-integer linear programming"),
            internal_algorithm_name="milp",
            website=f"{__DOC}scipy.optimize.milp.html",
        ),
    }

    def __init__(self, algo_name: str = "Scipy_MILP") -> None:  # noqa:D107
        super().__init__(algo_name)

    def _get_options(
        self,
        disp: bool = False,
        node_limit: int = 1000,
        presolve: bool = True,
        time_limit: int | None = None,
        mip_rel_gap: float = 0.0,
        **options: Any,
    ) -> dict[str, Any]:
        """Retrieve the options of the library.

        Define the default values for the options using the keyword arguments.

        Args:
            disp: Whether indicators of optimization status are to be printed to
                the console during optimization.
            node_limit: The maximum number of nodes (linear program relaxations) to
                solve before stopping.
            presolve: Whether to attempt to detect infeasibility,
                unboundedness or problem simplifications before solving.
                Refer to the SciPy documentation for more details.
            time_limit: The maximum number of seconds allotted to solve the problem.
                If ``None``, there is no time limit.
            mip_rel_gap: The termination criterion for MIP solver: solver will terminate
                when the gap between the primal objective value and the dual objective
                bound, scaled by the primal objective value, is <= mip_rel_gap.
            **options: The options for the algorithm.

        Returns:
            The processed options.
        """
        return self._process_options(
            disp=disp,
            node_limit=node_limit,
            presolve=presolve,
            time_limit=time_limit,
            mip_rel_gap=mip_rel_gap,
            **options,
        )

    def _run(self, problem: OptimizationProblem, **options: Any) -> OptimizationResult:
        # Remove the normalization option from the algorithm options
        options.pop(self._NORMALIZE_DESIGN_SPACE_OPTION, True)

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

        ineq_lhs, ineq_rhs = build_constraints_matrices(
            problem.constraints.get_originals(),
            MDOLinearFunction.ConstraintType.INEQ,
        )
        lq_constraints = []
        if ineq_lhs is not None:
            lq_constraints.append(
                LinearConstraint(
                    ineq_lhs,
                    -inf * ones_like(ineq_rhs),
                    ineq_rhs,
                    keep_feasible=True,
                )
            )

        eq_lhs, eq_rhs = build_constraints_matrices(
            problem.constraints.get_originals(),
            MDOLinearFunction.ConstraintType.EQ,
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
        # Pass the MILP to Scipy
        milp_result = milp(
            c=obj_coeff.real,
            bounds=bounds,
            constraints=lq_constraints,
            options=options,
            integrality=concatenate([
                problem.design_space.variable_types[variable_name]
                == problem.design_space.DesignVariableType.INTEGER
                for variable_name in problem.design_space.variable_names
            ]),
        )
        # Gather the optimization results
        x_opt = x_0 if milp_result.x is None else milp_result.x
        # N.B. SciPy tolerance on bounds is higher than the DesignSpace one
        x_opt = problem.design_space.project_into_bounds(x_opt)
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
        constraint_names = list(problem.constraints.original_to_current_names.keys())
        constraint_values = {key: val_opt[key] for key in constraint_names}
        constraints_grad = {key: jac_opt[key] for key in constraint_names}
        is_feasible = problem.constraints.is_point_feasible(val_opt)
        return OptimizationResult(
            x_0=x_0,
            x_opt=x_opt,
            f_opt=f_opt,
            status=milp_result.status,
            constraint_values=constraint_values,
            constraints_grad=constraints_grad,
            optimizer_name=self._algo_name,
            message=milp_result.message,
            n_obj_call=None,
            n_grad_call=None,
            n_constr_call=None,
            is_feasible=is_feasible,
        )
