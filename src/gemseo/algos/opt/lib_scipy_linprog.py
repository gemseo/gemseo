# -*- coding: utf-8 -*-
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
from typing import Any, Callable, Dict, Optional

from numpy import isfinite
from scipy.optimize import OptimizeResult, linprog

from gemseo.algos.opt.core.linear_constraints import build_constraints_matrices
from gemseo.algos.opt.opt_lib import OptimizationLibrary
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.algos.opt_result import OptimizationResult
from gemseo.core.mdofunctions.mdo_function import MDOLinearFunction
from gemseo.utils.py23_compat import PY2


class ScipyLinprog(OptimizationLibrary):
    """SciPy linear programming library interface.

    See OptimizationLibrary.
    """

    LIB_COMPUTE_GRAD = True

    REDUNDANCY_REMOVAL = "redundancy removal"
    REVISED_SIMPLEX = "REVISED_SIMPLEX"

    OPTIONS_MAP = {
        OptimizationLibrary.MAX_ITER: "maxiter",
        OptimizationLibrary.VERBOSE: "disp",
        REDUNDANCY_REMOVAL: "rr",
    }

    def __init__(self):
        """Constructor.

        Generate the library dictionary that contains the list of algorithms with their
        characteristics.
        """
        super(ScipyLinprog, self).__init__()
        doc = "https://docs.scipy.org/doc/scipy/reference/"

        self.lib_dict = {
            "LINEAR_INTERIOR_POINT": {
                self.INTERNAL_NAME: "interior-point",
                self.DESCRIPTION: "Linear programming by the interior-point"
                " method implemented in the SciPy library",
                self.WEBSITE: doc + "optimize.linprog-interior-point.html",
            },
            ScipyLinprog.REVISED_SIMPLEX: {
                self.INTERNAL_NAME: "revised simplex",
                self.DESCRIPTION: "Linear programming by a two-phase revised"
                " simplex algorithm implemented in the SciPy library",
                self.WEBSITE: doc + "optimize.linprog-revised_simplex.html",
            },
            "SIMPLEX": {
                self.INTERNAL_NAME: "simplex",
                self.DESCRIPTION: "Linear programming by the two-phase simplex"
                " algorithm implemented in the SciPy library",
                self.WEBSITE: doc + "optimize.linprog-simplex.html",
            },
        }
        if PY2:
            # The "revised simplex" algorithm is not available in Python 2.
            # Indeed this feature appeared in SciPy 1.3.0 which requires Python 3.5+.
            # https://docs.scipy.org/doc/scipy-1.3.0/reference/release.1.3.0.html
            del self.lib_dict["REVISED_SIMPLEX"]
        common_items = {
            self.PROBLEM_TYPE: OptimizationProblem.LINEAR_PB,
            self.POSITIVE_CONSTRAINTS: False,
            self.HANDLE_EQ_CONS: True,
            self.HANDLE_INEQ_CONS: True,
        }
        for algo_dict in self.lib_dict.values():
            algo_dict.update(common_items)

    def _get_options(
        self,
        max_iter=999,  # type: int
        autoscale=False,  # type: bool
        presolve=True,  # type: bool
        redundancy_removal=True,  # type: bool
        callback=None,  # type: Optional[Callable[[OptimizeResult], Any]]
        verbose=False,  # type: bool
        normalize_design_space=True,  # type: bool
        disp=False,  # type: bool
        **kwargs  # type: Any
    ):  # type: (...) -> Dict[str, Any]
        """Retrieve the options of the library.

        Define the default values for the options using the keyword arguments.

        Args:
            max_iter: The maximum number of iterations, i.e. unique calls to the
                objective function.
            autoscale: If True, then the linear problem is scaled.
                Refer to the SciPy documentation for more details.
            presolve: If True, then attempt to detect infeasibility,
                unboundedness or problem simplifications before solving.
                Refer to the SciPy documentation for more details.
            redundancy_removal: If True, then linearly dependent
                equality-constraints are removed.
            callback: A function to be called at least once per iteration.
                Takes a scipy.optimize.OptimizeResult as single argument.
                If None, no function is called.
                Refer to the SciPy documentation for more details.
            verbose: If True, then the convergence messages are printed.
            normalize_design_space: If True, scales variables in [0, 1].
            disp: Whether to print convergence messages.
            **kwargs: The other algorithm's options.

        Returns:
            The processed options.
        """
        normalize_ds = normalize_design_space
        options = self._process_options(
            max_iter=max_iter,
            autoscale=autoscale,
            presolve=presolve,
            redundancy_removal=redundancy_removal,
            verbose=verbose,
            callback=callback,
            normalize_design_space=normalize_ds,
            **kwargs
        )
        return options

    def _run(
        self, **options  # type: Any
    ):  # type: (...) -> OptimizationResult
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
        obj_coeff = self.problem.nonproc_objective.coefficients[0, :].real
        constraints = self.problem.nonproc_constraints
        ineq_lhs, ineq_rhs = build_constraints_matrices(
            constraints, MDOLinearFunction.TYPE_INEQ
        )
        eq_lhs, eq_rhs = build_constraints_matrices(
            constraints, MDOLinearFunction.TYPE_EQ
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
        linprog_result = linprog(
            c=obj_coeff,
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
        f_opt = val_opt[self.problem.objective.outvars[0]]
        constraints_values = {
            key: val_opt[key] for key in self.problem.get_constraints_names()
        }
        constraints_grad = {
            key: jac_opt[key] for key in self.problem.get_constraints_names()
        }
        is_feasible = self.problem.is_point_feasible(val_opt)
        optim_result = OptimizationResult(
            x_0=x_0,
            x_opt=x_opt,
            f_opt=f_opt,
            status=linprog_result.status,
            constraints_values=constraints_values,
            constraints_grad=constraints_grad,
            optimizer_name=self.algo_name,
            message=linprog_result.message,
            n_obj_call=None,
            n_grad_call=None,
            n_constr_call=None,
            is_feasible=is_feasible,
        )

        return optim_result
