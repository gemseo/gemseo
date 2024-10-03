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
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Final

from numpy import promote_types
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import bicg
from scipy.sparse.linalg import bicgstab
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import cgs
from scipy.sparse.linalg import gcrotmk
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import lgmres
from scipy.sparse.linalg import splu
from scipy.sparse.linalg import tfqmr

from gemseo.algos.linear_solvers.base_linear_solver_library import (
    BaseLinearSolverLibrary,
)
from gemseo.algos.linear_solvers.base_linear_solver_library import (
    LinearSolverDescription,
)
from gemseo.algos.linear_solvers.base_linear_solver_settings import (
    LinearSolverLibrarySettings,
)
from gemseo.algos.linear_solvers.scipy_linalg._base_scipy_linalg_settings import (
    BaseSciPyLinalgSettings,
)
from gemseo.algos.linear_solvers.scipy_linalg.settings.bicg import BICGSettings
from gemseo.algos.linear_solvers.scipy_linalg.settings.bicgstab import BICGStabSettings
from gemseo.algos.linear_solvers.scipy_linalg.settings.cg import CGSettings
from gemseo.algos.linear_solvers.scipy_linalg.settings.cgs import CGSSettings
from gemseo.algos.linear_solvers.scipy_linalg.settings.gcrot import GCROTSettings
from gemseo.algos.linear_solvers.scipy_linalg.settings.gmres import GMRESSettings
from gemseo.algos.linear_solvers.scipy_linalg.settings.lgmres import LGMRESSettings
from gemseo.utils.compatibility.scipy import TOL_OPTION
from gemseo.utils.compatibility.scipy import array_classes

if TYPE_CHECKING:
    from gemseo.algos.linear_solvers.linear_problem import LinearProblem
    from gemseo.typing import NumberArray
    from gemseo.typing import SparseOrDenseRealArray
    from gemseo.typing import StrKeyMapping

LOGGER = logging.getLogger(__name__)


@dataclass
class ScipyLinAlgAlgorithmDescription(LinearSolverDescription):
    """The description of the SciPy linear algebra library."""

    library_name: str = "SciPy Linear Algebra"
    """The library name."""

    settings: type[BaseSciPyLinalgSettings] = BaseSciPyLinalgSettings
    """The option validation model for SciPy linear algebra library."""


class ScipyLinalgAlgos(BaseLinearSolverLibrary):
    """Wrapper for scipy linalg sparse linear solvers."""

    file_path: str
    """The path to the file where the problem is saved when it is not converged.

    This will be set only if the optio, `save_when_fail` is set to `True`.
    """

    __BASE_INFO_MSG: ClassVar[str] = "SciPy linear solver algorithm stop info: "

    __NAMES_TO_FUNCTIONS: ClassVar[dict[str, Callable]] = {
        "BICG": bicg,
        "BICGSTAB": bicgstab,
        "CG": cg,
        "CGS": cgs,
        "GMRES": gmres,
        "LGMRES": lgmres,
        "GCROT": gcrotmk,
        "TFQMR": tfqmr,
        "DEFAULT": lgmres,
    }
    """The algorithm name bound to the SciPy function."""

    __DOC: Final[str] = "https://docs.scipy.org/doc/scipy/reference/"

    ALGORITHM_INFOS: ClassVar[dict[str, LinearSolverDescription]] = {
        "BICG": LinearSolverDescription(
            algorithm_name="BICG",
            description="BI-Conjugate Gradient",
            internal_algorithm_name="bicg",
            website=f"{__DOC}generated/scipy.sparse.linalg.bicg.html",
            settings=BICGSettings,
        ),
        "BICGSTAB": LinearSolverDescription(
            algorithm_name="BICGSTAB",
            description="Bi-Conjugate Gradient STABilized",
            internal_algorithm_name="bicgstab",
            website=f"{__DOC}generated/scipy.sparse.linalg.bicgstab.html",
            settings=BICGStabSettings,
        ),
        "CG": LinearSolverDescription(
            algorithm_name="CG",
            description="Conjugate Gradient",
            internal_algorithm_name="cg",
            lhs_must_be_symmetric=True,
            lhs_must_be_positive_definite=True,
            website=f"{__DOC}generated/scipy.sparse.linalg.cg.html",
            settings=CGSettings,
        ),
        "CGS": LinearSolverDescription(
            algorithm_name="CGS",
            description="Conjugate Gradient Squared",
            internal_algorithm_name="cgs",
            website=f"{__DOC}generated/scipy.sparse.linalg.cgs.html",
            settings=CGSSettings,
        ),
        "GCROT": LinearSolverDescription(
            algorithm_name="GCROT",
            description="Generalized Conjugate Residual with Optimal Truncation",
            internal_algorithm_name="gcrotmk",
            website=f"{__DOC}generated/scipy.sparse.linalg.gcrotmk.html",
            settings=GCROTSettings,
        ),
        "GMRES": LinearSolverDescription(
            algorithm_name="GMRES",
            description="Generalized Minimum RESidual",
            internal_algorithm_name="gmres",
            website=f"{__DOC}generated/scipy.sparse.linalg.gmres.html",
            settings=GMRESSettings,
        ),
        "LGMRES": LinearSolverDescription(
            algorithm_name="LGMRES",
            description="Loose Generalized Minimum RESidual",
            internal_algorithm_name="lgmres",
            website=f"{__DOC}generated/scipy.sparse.linalg.lgmres.html",
            settings=LGMRESSettings,
        ),
        "TFQMR": LinearSolverDescription(
            algorithm_name="TFQMR",
            description="Transpose-Free Quasi-Minimal Residual",
            internal_algorithm_name="tfqmr",
            website=f"{__DOC}generated/scipy.sparse.linalg.tfqmr.html",
            settings=BaseSciPyLinalgSettings,
        ),
        "DEFAULT": LinearSolverDescription(
            algorithm_name="DEFAULT",
            description="Default solver (LGMRES)",
            internal_algorithm_name="lgmres",
            website=f"{__DOC}generated/scipy.sparse.linalg.lgmres.html",
            settings=LGMRESSettings,
        ),
    }

    def _pre_run(self, problem: LinearProblem, **options: Any) -> None:
        if options["use_ilu_precond"] and options["M"] is not None:
            msg = (
                "Use either 'use_ilu_precond' or "
                "provide 'preconditioner', but not both."
            )
            raise ValueError(msg)

        if issparse(problem.rhs):
            problem.rhs = problem.rhs.toarray()

        rhs = problem.rhs
        lhs = problem.lhs

        c_dtype = None
        if rhs.dtype != lhs.dtype and not isinstance(lhs, LinearOperator):
            c_dtype = promote_types(rhs.dtype, lhs.dtype)
            if lhs.dtype != c_dtype:
                problem.lhs = lhs.astype(c_dtype)
            if rhs.dtype != c_dtype:
                problem.rhs = rhs.astype(c_dtype)

        super()._pre_run(problem, **options)

    def _run(self, problem: LinearProblem, **options: Any) -> NumberArray:
        if options["use_ilu_precond"] and not isinstance(problem.lhs, LinearOperator):
            options["M"] = self._build_ilu_preconditioner(problem.lhs)

        if options["store_residuals"]:
            options["callback"] = self.__store_residuals

        options_ = self._filter_settings(
            options, model_to_exclude=LinearSolverLibrarySettings
        )

        linear_solver = self.__NAMES_TO_FUNCTIONS[self._algo_name]
        problem.solution, info = linear_solver(problem.lhs, problem.rhs, **options_)
        self._check_solver_info(info, options_)

        return problem.solution

    def __store_residuals(self, current_x: NumberArray) -> NumberArray:
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
        rhs: NumberArray,
        **options: Any,
    ) -> tuple[NumberArray, int]:
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