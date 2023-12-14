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
from typing import Any
from typing import ClassVar

from numpy import concatenate
from numpy import inf
from numpy import ones_like
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import milp

from gemseo.algos.opt.core.linear_constraints import build_constraints_matrices
from gemseo.algos.opt.optimization_library import OptimizationAlgorithmDescription
from gemseo.algos.opt.optimization_library import OptimizationLibrary
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.algos.opt_result import OptimizationResult
from gemseo.core.mdofunctions.mdo_linear_function import MDOLinearFunction
from gemseo.utils.compatibility.scipy import sparse_classes


@dataclass
class ScipyMILPAlgorithmDescription(OptimizationAlgorithmDescription):
    """The description of a MILP optimization algorithm from the SciPy library."""

    problem_type: OptimizationProblem.ProblemType = (
        OptimizationProblem.ProblemType.LINEAR
    )
    handle_equality_constraints: bool = True
    handle_inequality_constraints: bool = True
    library_name: str = "SciPy"
    handle_integer_variables: bool = True


class ScipyMILP(OptimizationLibrary):
    """SciPy Mixed Integer Linear Programming library interface.

    See OptimizationLibrary.

    With respect to scipy milp function, this wrapper only allows continuous or integer
    variables.
    """

    LIB_COMPUTE_GRAD = True

    OPTIONS_MAP: ClassVar[dict[int, str]] = {
        OptimizationLibrary.MAX_ITER: "node_limit",
    }

    LIBRARY_NAME = "SciPy"

    _SUPPORT_SPARSE_JACOBIAN: ClassVar[bool] = True
    """Whether the library support sparse Jacobians."""

    def __init__(self) -> None:
        """Constructor.

        Generate the library dictionary that contains the list of algorithms with their
        characteristics.
        """
        super().__init__()
        doc = "https://docs.scipy.org/doc/scipy/reference/"
        self.algo_name = "Scipy_MILP"
        self.descriptions = {
            "Scipy_MILP": ScipyMILPAlgorithmDescription(
                algorithm_name="Branch & Cut algorithm",
                description=("Mixed-integer linear programming"),
                internal_algorithm_name="milp",
                website=f"{doc}scipy.optimize.milp.html",
            ),
        }

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

    def _run(self, **options: Any) -> OptimizationResult:
        # Remove the normalization option from the algorithm options
        options.pop(self.NORMALIZE_DESIGN_SPACE_OPTION, True)

        # Get the starting point and bounds
        x_0, l_b, u_b = self.get_x0_and_bounds_vects(False)
        # Replace infinite bounds with None
        bounds = Bounds(lb=l_b, ub=u_b, keep_feasible=True)

        # Build the functions matrices
        # N.B. use the non-processed functions to access the coefficients
        coefficients = self.problem.nonproc_objective.coefficients
        if isinstance(coefficients, sparse_classes):
            obj_coeff = coefficients.getrow(0).todense().flatten()
        else:
            obj_coeff = coefficients[0, :]

        constraints = self.problem.nonproc_constraints
        ineq_lhs, ineq_rhs = build_constraints_matrices(
            constraints, MDOLinearFunction.ConstraintType.INEQ
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
            constraints, MDOLinearFunction.ConstraintType.EQ
        )
        if eq_lhs is not None:
            lq_constraints.append(
                LinearConstraint(
                    eq_lhs,
                    eq_rhs - self.problem.eq_tolerance,
                    eq_rhs + self.problem.eq_tolerance,
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
                self.problem.design_space.variable_types[variable_name]
                == self.problem.design_space.DesignVariableType.INTEGER
                for variable_name in self.problem.design_space.variable_names
            ]),
        )
        # Gather the optimization results
        x_opt = x_0 if milp_result.x is None else milp_result.x
        # N.B. SciPy tolerance on bounds is higher than the DesignSpace one
        x_opt = self.problem.design_space.project_into_bounds(x_opt)
        val_opt, jac_opt = self.problem.evaluate_functions(
            x_vect=x_opt,
            eval_jac=True,
            eval_obj=True,
            normalize=False,
            no_db_no_norm=True,
        )
        f_opt = val_opt[self.problem.objective.name]
        constraint_names = list(self.problem.constraint_names.keys())
        constraint_values = {key: val_opt[key] for key in constraint_names}
        constraints_grad = {key: jac_opt[key] for key in constraint_names}
        is_feasible = self.problem.is_point_feasible(val_opt)
        return OptimizationResult(
            x_0=x_0,
            x_opt=x_opt,
            f_opt=f_opt,
            status=milp_result.status,
            constraint_values=constraint_values,
            constraints_grad=constraints_grad,
            optimizer_name=self.algo_name,
            message=milp_result.message,
            n_obj_call=None,
            n_grad_call=None,
            n_constr_call=None,
            is_feasible=is_feasible,
        )
