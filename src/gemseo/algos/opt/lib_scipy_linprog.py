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
from typing import Callable
from typing import ClassVar
from typing import Final

from numpy import isfinite
from scipy.optimize import OptimizeResult
from scipy.optimize import linprog

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
class ScipyLinProgAlgorithmDescription(OptimizationAlgorithmDescription):
    """The description of a linear optimization algorithm from the SciPy library."""

    for_linear_problems: bool = True
    handle_equality_constraints: bool = True
    handle_inequality_constraints: bool = True
    library_name: str = "SciPy"


class ScipyLinprog(BaseOptimizationLibrary):
    """SciPy linear programming library interface.

    See BaseOptimizationLibrary.
    """

    _REDUNDANCY_REMOVAL: Final[str] = "redundancy removal"

    _OPTIONS_MAP: ClassVar[dict[Any, str]] = {
        BaseOptimizationLibrary._MAX_ITER: "maxiter",
        BaseOptimizationLibrary._VERBOSE: "disp",
        _REDUNDANCY_REMOVAL: "rr",
    }

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
        ),
        "HIGHS_DUAL_SIMPLEX": ScipyLinProgAlgorithmDescription(
            algorithm_name="Dual simplex",
            description=("Linear programming using the HiGHS dual simplex solver."),
            internal_algorithm_name="highs-ds",
            website=f"{__DOC}optimize.linprog-highs-ds.html",
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

    def _get_options(
        self,
        max_iter: int = 999,
        autoscale: bool = False,
        presolve: bool = True,
        redundancy_removal: bool = True,
        callback: Callable[[OptimizeResult], Any] | None = None,
        verbose: bool = False,
        normalize_design_space: bool = True,
        disp: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Retrieve the options of the library.

        Define the default values for the options using the keyword arguments.

        Args:
            max_iter: The maximum number of iterations, i.e. unique calls to the
                objective function.
            autoscale: If ``True``, then the linear problem is scaled.
                Refer to the SciPy documentation for more details.
            presolve: If ``True``, then attempt to detect infeasibility,
                unboundedness or problem simplifications before solving.
                Refer to the SciPy documentation for more details.
            redundancy_removal: If ``True``, then linearly dependent
                equality-constraints are removed.
            callback: A function to be called at least once per iteration.
                Takes a scipy.optimize.OptimizeResult as single argument.
                If ``None``, no function is called.
                Refer to the SciPy documentation for more details.
            verbose: If ``True``, then the convergence messages are printed.
            normalize_design_space: If ``True``, scales variables in [0, 1].
            disp: Whether to print convergence messages.
            **kwargs: The other algorithm's options.

        Returns:
            The processed options.
        """
        return self._process_options(
            max_iter=max_iter,
            autoscale=autoscale,
            presolve=presolve,
            redundancy_removal=redundancy_removal,
            verbose=verbose,
            callback=callback,
            normalize_design_space=normalize_design_space,
            disp=disp,
            **kwargs,
        )

    def _run(self, problem: OptimizationProblem, **options: Any) -> OptimizationResult:
        # Remove the normalization option from the algorithm options
        options.pop(self._NORMALIZE_DESIGN_SPACE_OPTION, True)

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

        # |g| is in charge of ensuring max iterations, since it may
        # have a different definition of iterations, such as for SLSQP
        # for instance which counts duplicate calls to x as a new iteration
        options[self._OPTIONS_MAP[self._MAX_ITER]] = 10000000

        # For the "revised simplex" algorithm (available since SciPy 1.3.0 which
        # requires Python 3.5+) the initial guess must be a basic feasible solution,
        # or BFS (geometrically speaking, a vertex of the feasible polyhedron).
        # Here the passed initial guess is always ignored.
        # (A BFS will be automatically looked for during the first phase of the simplex
        # algorithm.)
        # TODO: interface the option ``integrality`` of HiGHS solvers
        # to support mixed-integer linear programming
        linprog_result = linprog(
            c=obj_coeff.real,
            A_ub=ineq_lhs,
            b_ub=ineq_rhs,
            A_eq=eq_lhs,
            b_eq=eq_rhs,
            bounds=bounds,
            method=self.ALGORITHM_INFOS[self._algo_name].internal_algorithm_name,
            options=options,
        )

        # Gather the optimization results
        x_opt = linprog_result.x
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
        constraint_values = {
            key: val_opt[key] for key in problem.constraints.get_names()
        }
        constraints_grad = {
            key: jac_opt[key] for key in problem.constraints.get_names()
        }
        is_feasible = problem.constraints.is_point_feasible(val_opt)
        return OptimizationResult(
            x_0=x_0,
            x_0_as_dict=problem.design_space.array_to_dict(x_0),
            x_opt=x_opt,
            x_opt_as_dict=problem.design_space.array_to_dict(x_opt),
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
