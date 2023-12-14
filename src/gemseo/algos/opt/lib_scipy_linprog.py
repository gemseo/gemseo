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
from typing import Any
from typing import Callable
from typing import ClassVar

from numpy import isfinite
from scipy.optimize import OptimizeResult
from scipy.optimize import linprog

from gemseo.algos.opt.core.linear_constraints import build_constraints_matrices
from gemseo.algos.opt.optimization_library import OptimizationAlgorithmDescription
from gemseo.algos.opt.optimization_library import OptimizationLibrary
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.algos.opt_result import OptimizationResult
from gemseo.core.mdofunctions.mdo_linear_function import MDOLinearFunction
from gemseo.utils.compatibility.scipy import sparse_classes


@dataclass
class ScipyLinProgAlgorithmDescription(OptimizationAlgorithmDescription):
    """The description of a linear optimization algorithm from the SciPy library."""

    problem_type: OptimizationProblem.ProblemType = (
        OptimizationProblem.ProblemType.LINEAR
    )
    handle_equality_constraints: bool = True
    handle_inequality_constraints: bool = True
    library_name: str = "SciPy"


class ScipyLinprog(OptimizationLibrary):
    """SciPy linear programming library interface.

    See OptimizationLibrary.
    """

    LIB_COMPUTE_GRAD = True

    REDUNDANCY_REMOVAL = "redundancy removal"

    REVISED_SIMPLEX = "REVISED_SIMPLEX"

    OPTIONS_MAP: ClassVar[dict[Any, str]] = {
        OptimizationLibrary.MAX_ITER: "maxiter",
        OptimizationLibrary.VERBOSE: "disp",
        REDUNDANCY_REMOVAL: "rr",
    }

    LIBRARY_NAME = "SciPy"

    _SUPPORT_SPARSE_JACOBIAN: ClassVar[bool] = True
    """Whether the library supports sparse Jacobians."""

    def __init__(self) -> None:
        """Constructor.

        Generate the library dictionary that contains the list of algorithms with their
        characteristics.
        """
        super().__init__()
        doc = "https://docs.scipy.org/doc/scipy/reference/"

        # TODO: Remove legacy methods "interior-point", "revised simplex" and "simplex".
        self.descriptions = {
            "LINEAR_INTERIOR_POINT": ScipyLinProgAlgorithmDescription(
                algorithm_name="Linear interior point",
                description=(
                    "Linear programming by the interior-point"
                    " method implemented in the SciPy library"
                ),
                internal_algorithm_name="interior-point",
                website=f"{doc}optimize.linprog-interior-point.html",
            ),
            ScipyLinprog.REVISED_SIMPLEX: ScipyLinProgAlgorithmDescription(
                algorithm_name="Revised simplex",
                description=(
                    "Linear programming by a two-phase revised"
                    " simplex algorithm implemented in the SciPy library"
                ),
                internal_algorithm_name="revised simplex",
                website=f"{doc}optimize.linprog-revised_simplex.html",
            ),
            "SIMPLEX": ScipyLinProgAlgorithmDescription(
                algorithm_name="Simplex",
                description=(
                    "Linear programming by the two-phase simplex"
                    " algorithm implemented in the SciPy library"
                ),
                internal_algorithm_name="simplex",
                website=f"{doc}optimize.linprog-simplex.html",
            ),
            "HIGHS_INTERIOR_POINT": ScipyLinProgAlgorithmDescription(
                algorithm_name="Interior point method",
                description=(
                    "Linear programming using the HiGHS interior point solver."
                ),
                internal_algorithm_name="highs-ipm",
                website=f"{doc}optimize.linprog-highs-ipm.html",
            ),
            "HIGHS_DUAL_SIMPLEX": ScipyLinProgAlgorithmDescription(
                algorithm_name="Dual simplex",
                description=("Linear programming using the HiGHS dual simplex solver."),
                internal_algorithm_name="highs-ds",
                website=f"{doc}optimize.linprog-highs-ds.html",
            ),
            "HIGHS": ScipyLinProgAlgorithmDescription(
                algorithm_name="HiGHS",
                description=(
                    "Linear programming using the HiGHS solvers. "
                    "A choice is automatically made between the dual simplex "
                    "and the interior-point method."
                ),
                internal_algorithm_name="highs",
                website=f"{doc}optimize.linprog-highs.html",
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

    def _run(self, **options: Any) -> OptimizationResult:
        """Run the algorithm.

        Args:
            **options: options dictionary for the algorithm.

        Returns:
            The optimization result.
        """
        # Remove the normalization option from the algorithm options
        options.pop(self.NORMALIZE_DESIGN_SPACE_OPTION, True)

        # Get the starting point and bounds
        x_0, l_b, u_b = self.get_x0_and_bounds_vects(False)
        # Replace infinite bounds with None
        l_b = [val if isfinite(val) else None for val in l_b]
        u_b = [val if isfinite(val) else None for val in u_b]
        bounds = list(zip(l_b, u_b))

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
        eq_lhs, eq_rhs = build_constraints_matrices(
            constraints, MDOLinearFunction.ConstraintType.EQ
        )

        # |g| is in charge of ensuring max iterations, since it may
        # have a different definition of iterations, such as for SLSQP
        # for instance which counts duplicate calls to x as a new iteration
        options[self.OPTIONS_MAP[self.MAX_ITER]] = 10000000

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
            method=self.internal_algo_name,
            options=options,
        )

        # Gather the optimization results
        x_opt = linprog_result.x
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
        constraint_values = {
            key: val_opt[key] for key in self.problem.get_constraint_names()
        }
        constraints_grad = {
            key: jac_opt[key] for key in self.problem.get_constraint_names()
        }
        is_feasible = self.problem.is_point_feasible(val_opt)
        return OptimizationResult(
            x_0=x_0,
            x_0_as_dict=self.problem.design_space.array_to_dict(x_0),
            x_opt=x_opt,
            x_opt_as_dict=self.problem.design_space.array_to_dict(x_opt),
            f_opt=f_opt,
            objective_name=self.problem.objective.name,
            status=linprog_result.status,
            constraint_values=constraint_values,
            constraints_grad=constraints_grad,
            optimizer_name=self.algo_name,
            message=linprog_result.message,
            n_obj_call=None,
            n_grad_call=None,
            n_constr_call=None,
            is_feasible=is_feasible,
        )
