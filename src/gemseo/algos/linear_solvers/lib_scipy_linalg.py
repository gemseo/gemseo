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
"""Wrappers for SciPy's linear solvers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import ClassVar

from numpy import ndarray
from numpy import promote_types
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import bicg
from scipy.sparse.linalg import bicgstab
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import lgmres
from scipy.sparse.linalg import qmr
from scipy.sparse.linalg import splu

from gemseo.algos.linear_solvers.base_linear_solver_library import (
    BaseLinearSolverLibrary,
)
from gemseo.algos.linear_solvers.base_linear_solver_library import (
    LinearSolverDescription,
)
from gemseo.utils.compatibility.scipy import SCIPY_LOWER_THAN_1_12
from gemseo.utils.compatibility.scipy import TOL_OPTION
from gemseo.utils.compatibility.scipy import array_classes

if TYPE_CHECKING:
    from gemseo.algos.linear_solvers.linear_problem import LinearProblem
    from gemseo.typing import SparseOrDenseRealArray
    from gemseo.typing import StrKeyMapping

LOGGER = logging.getLogger(__name__)


class ScipyLinalgAlgos(BaseLinearSolverLibrary):
    """Wrapper for scipy linalg sparse linear solvers."""

    file_path: str
    """The path to the file to saved the problem when it is not converged and the option
    save_when_fail is active."""

    methods_map: dict[str, Callable[[ndarray, ndarray, ...], tuple[ndarray, int]]]
    """The mapping between the solver names and the solvers methods in scipy.sparse."""

    __BASE_INFO_MSG: ClassVar[str] = "scipy linear solver algorithm stop info: "
    _OPTIONS_MAP: ClassVar[dict[str, str]] = {
        "max_iter": "maxiter",
        "preconditioner": "M",
        "store_outer_av": "store_outer_Av",
    }

    _LGMRES_SPEC_OPTS: ClassVar[tuple[str, str, str, str, str]] = (
        "inner_m",
        "outer_k",
        "outer_v",
        "store_outer_av",
        "prepend_outer_v",
    )

    methods_map: ClassVar[dict[str, Callable]] = {
        "LGMRES": lgmres,
        "GMRES": gmres,
        "BICG": bicg,
        "QMR": qmr,
        "BICGSTAB": bicgstab,
        "DEFAULT": "None",
    }

    ALGORITHM_INFOS: ClassVar[dict[str, LinearSolverDescription]] = {
        algo_name: LinearSolverDescription(
            algorithm_name=algo_name,
            description="Linear solver implemented in the SciPy library.",
            internal_algorithm_name=algo_name,
            lhs_must_be_linear_operator=True,
            library_name="SciPy",
            website=f"https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.{algo_name.lower()}.html",
        )
        for algo_name in methods_map
    }
    ALGORITHM_INFOS["DEFAULT"].website = (
        "https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html",
    )
    ALGORITHM_INFOS["DEFAULT"].description = (
        "This starts by LGMRES, but if it fails, "
        "switches to GMRES, then direct method super LU factorization."
    )

    def _get_options(
        self,
        max_iter: int = 1000,
        preconditioner: ndarray | LinearOperator | None = None,
        rtol: float = 1e-12,
        atol: float = 1e-12,
        x0: ndarray | None = None,
        use_ilu_precond: bool = True,
        inner_m: int = 30,
        outer_k: int = 3,
        outer_v: list[tuple] | None = None,
        store_outer_av: bool = True,
        prepend_outer_v: bool = False,
        save_when_fail: bool = False,
        store_residuals: bool = False,
    ) -> dict[str, Any]:
        """Check the options and set the default values.

        See https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html

        Args:
            max_iter: The maximum number of iterations.
            preconditioner: The preconditioner, approximation of RHS^-1.
                If ``None``, no preconditioner is used.
            rtol: The relative tolerance for convergence;
                ``norm(b - A @ x) <= max(rtol*norm(b), atol)`` should be satisfied.
            atol: The absolute tolerance for convergence;
                ``norm(b - A @ x) <= max(rtol*norm(b), atol)`` should be satisfied.
            x0: The initial guess for the solution.
                M{sparse matrix, dense matrix, LinearOperator}.
                If ``None``, solvers usually start from the null vector.
            inner_m: The number of inner GMRES iterations per outer iteration.
            outer_k: The number of vectors to carry between inner GMRES iterations.
            outer_v:  The data used to augment the Krylov subspace.
            store_outer_av: Whether LGMRES should store also A*v in
                addition to the vectors v in outer_v.
            prepend_outer_v: Whether to put outer_v
                augmentation vectors before the Krylov iterates.
            use_ilu_precond: Whether to use superLU to
                compute an incomplete LU factorization as preconditioner.
            save_when_fail: Whether to save the linear system to the disk
                when the solver failed to converge.
            store_residuals: Whether to store the residuals convergence history.

        Returns:
            The options.

        Raises:
            ValueError: When the LHR and RHS shapes are inconsistent, or
                when the preconditioner options are inconsistent.
        """
        if preconditioner is not None and (
            preconditioner.shape != self._problem.lhs.shape
        ):
            msg = "Inconsistent Preconditioner shape: %s != %s"
            raise ValueError(msg.format(preconditioner.shape, self._problem.lhs.shape))

        if x0 is not None and len(x0) != len(self._problem.rhs):
            msg = "Inconsistent initial guess shape: %s != %s"
            raise ValueError(msg.format(x0.shape, self._problem.rhs.shape))

        if use_ilu_precond and preconditioner is not None:
            msg = (
                "Use either 'use_ilu_precond' or provide 'preconditioner',"
                " but not both."
            )
            raise ValueError(msg)

        return self._process_options(
            max_iter=max_iter,
            preconditioner=preconditioner,
            rtol=rtol,
            atol=atol,
            x0=x0,
            inner_m=inner_m,
            outer_k=outer_k,
            outer_v=outer_v,
            store_outer_av=store_outer_av,
            prepend_outer_v=prepend_outer_v,
            use_ilu_precond=use_ilu_precond,
            save_when_fail=save_when_fail,
            store_residuals=store_residuals,
        )

    def _run(
        self, problem: LinearProblem, **options: None | bool | float | ndarray
    ) -> ndarray:
        if SCIPY_LOWER_THAN_1_12:
            options["tol"] = options.pop("rtol")

        if issparse(problem.rhs):
            problem.rhs = problem.rhs.toarray()
        rhs = problem.rhs
        lhs = problem.lhs

        opts_solver = options.copy()
        c_dtype = None

        if rhs.dtype != lhs.dtype and not isinstance(lhs, LinearOperator):
            c_dtype = promote_types(rhs.dtype, lhs.dtype)
            if lhs.dtype != c_dtype:
                lhs = lhs.astype(c_dtype)
            if rhs.dtype != c_dtype:
                rhs = rhs.astype(c_dtype)

        if opts_solver["use_ilu_precond"] and not isinstance(lhs, LinearOperator):
            opts_solver["M"] = self._build_ilu_preconditioner(lhs, c_dtype)

        del opts_solver["use_ilu_precond"]
        del opts_solver["store_residuals"]

        if self._algo_name == "DEFAULT":
            method = self._run_default_solver
        else:
            method = self.methods_map[self._algo_name]

        if options.get("store_residuals", False):
            opts_solver["callback"] = self.__store_residuals

        opts_solver.pop("save_when_fail")
        problem.solution, info = method(lhs, rhs, **opts_solver)
        self._check_solver_info(info, options)

        return problem.solution

    def __store_residuals(self, current_x: ndarray) -> ndarray:
        """Store the current iteration residuals.

        Args:
            current_x: The current solution.
        """
        self._problem.solution = current_x
        self._problem.compute_residuals(True, True)

    def _check_solver_info(
        self,
        info: int,
        options: StrKeyMapping,
    ) -> bool:
        """Check the info returned by the solver.

        Args:
            info: The info value, negative, 0 or positive depending
                on status.
            options: The options passed to the solver.

        Returns:
            Whether the solver converged.

        Raises:
            RuntimeError: If the inputs are illegal for the solver.
        """
        self._problem.is_converged = info == 0

        if info > 0:
            if self._problem.solution is not None:
                LOGGER.warning(
                    "%s, residual = %s",
                    self.__BASE_INFO_MSG,
                    self._problem.compute_residuals(True),
                )
                LOGGER.warning("info = %s", info)
            return False

        # check the dimensions
        if info < 0:
            raise RuntimeError(
                self.__BASE_INFO_MSG + "illegal input or breakdown, options = %s",
                options,
            )

        return True

    def _run_default_solver(
        self,
        lhs: SparseOrDenseRealArray | LinearOperator,
        rhs: ndarray,
        **options: Any,
    ) -> tuple[ndarray, int]:
        """Run the default solver.

        This starts by LGMRES, but if it fails, switches to GMRES,
        then direct method super LU factorization.

        Args:
            lhs: The left hand side of the equation (matrix).
            rhs: The right hand side of the equation.
            **options: The user options.

        Returns:
            The last solution found and the info.
        """
        # Try LGMRES first
        best_sol, info = lgmres(A=lhs, b=rhs, **options)
        best_res = self._problem.compute_residuals(True, current_x=best_sol)

        if self._check_solver_info(info, options):
            return best_sol, info

        # If not converged, try GMRES

        # Adapt options
        for k in self._LGMRES_SPEC_OPTS:
            if k in options:
                del options[k]

        if best_res < 1.0:
            options["x0"] = best_sol

        sol, info = gmres(A=lhs, b=rhs, **options)
        res = self._problem.compute_residuals(True, current_x=sol)

        if res < best_res:
            best_sol = sol
            options["x0"] = best_sol

        if self._check_solver_info(info, options):  # pragma: no cover
            return best_sol, info

        info = 1
        self._problem.is_converged = False

        # Attempt direct solver when possible
        if isinstance(lhs, array_classes):
            a_fact = splu(lhs)
            sol = a_fact.solve(rhs)
            res = self._problem.compute_residuals(True, current_x=sol)

            if res < options[TOL_OPTION]:  # pragma: no cover
                best_sol = sol
                info = 0
                self._problem.is_converged = True

        return best_sol, info
