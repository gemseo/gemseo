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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

"""Wrappers for scipy's linear solvers."""

from __future__ import division, unicode_literals

import logging
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from numpy import find_common_type, ndarray
from scipy.sparse import spmatrix
from scipy.sparse.base import issparse
from scipy.sparse.linalg import LinearOperator, bicg, bicgstab, gmres, lgmres, qmr, splu

from gemseo.algos.linear_solvers.linear_solver_lib import LinearSolverLib

LOGGER = logging.getLogger(__name__)


class ScipyLinalgAlgos(LinearSolverLib):
    """Wrapper for scipy linalg sparse linear solvers.

    Attributes:
       save_fpath (str): The path to the file to saved the problem when
           it is not converged and the option save_when_fail
           is active.

       methods_map (Dict[str, Callable]): The mapping between the solver names
           and the solvers methods in scipy.sparse.
    """

    BASE_INFO_MSG = "scipy linear solver algorithm stop info: "
    OPTIONS_MAP = {
        "max_iter": "maxiter",
        "preconditioner": "M",
        "store_outer_av": "store_outer_Av",
    }

    LGMRES_SPEC_OPTS = (
        "inner_m",
        "outer_k",
        "outer_v",
        "store_outer_av",
        "prepend_outer_v",
    )

    __WEBSITE = "https://docs.scipy.org/doc/scipy/reference/generated/{}.html"
    __WEBPAGE = "scipy.sparse.linalg.{}"
    __WEBPAGES = {
        "BICG": __WEBPAGE.format("bicg"),
        "GMRES": __WEBPAGE.format("gmres"),
        "LGMRES": __WEBPAGE.format("lgmres"),
        "QMR": __WEBPAGE.format("qmr"),
        "BICGSTAB": __WEBPAGE.format("bicgstab"),
        "DEFAULT": __WEBPAGE.format("splu"),
    }

    def __init__(self):  # type: (...) -> None
        super(ScipyLinalgAlgos, self).__init__()
        self.methods_map = {
            "LGMRES": lgmres,
            "GMRES": gmres,
            "BICG": bicg,
            "QMR": qmr,
            "BICGSTAB": bicgstab,
            "DEFAULT": self._run_default_solver,
        }
        self.lib_dict = {
            name: self.get_default_properties(name) for name in self.methods_map.keys()
        }
        self.lib_dict["DEFAULT"]["description"] = (
            "This starts by LGMRES, but if it fails, "
            "switches to GMRES, then direct method super LU factorization."
        )

    @classmethod
    def get_default_properties(
        cls, algo_name  # type: str
    ):  # type: (...) -> Dict[str, Union[bool,str]]
        """Return the properties of the algorithm.

        It states if it requires symmetric,
        or positive definite matrices for instance.

        Args:
            algo_name: The algorithm name.

        Returns:
            The properties of the solver.
        """
        return {
            cls.LHS_MUST_BE_POSITIVE_DEFINITE: False,
            cls.LHS_MUST_BE_SYMMETRIC: False,
            cls.LHS_CAN_BE_LINEAR_OPERATOR: True,
            cls.INTERNAL_NAME: algo_name,
            "description": "Linear solver implemented in the SciPy library.",
            "website": cls.__WEBSITE.format(algo_name),
        }

    def _get_options(
        self,
        max_iter=1000,  # type: int
        preconditioner=None,  # type: Optional[Union[ndarray, LinearOperator]]
        tol=1e-12,  # type: float
        atol=None,  # type: Optional[float]
        x0=None,  # type: Optional[ndarray]
        use_ilu_precond=True,  # type: bool
        inner_m=30,  # type: int
        outer_k=3,  # type: int
        outer_v=None,  # type: Optional[List[Tuple]]
        store_outer_av=True,  # type: bool
        prepend_outer_v=False,  # type: bool
        save_when_fail=False,  # type: bool
        store_residuals=False,  # type: bool
    ):  # type: (...) -> Dict[str, Any]
        """Check the options and sets the default values.

        See https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html

        Args:
            max_iter: The maximum number of iterations.
            preconditioner: The preconditionner, approximation of RHS^-1.
                If None, no preconditioner is used.
            tol: The relative tolerance for convergence,
                norm(RHS.dot(sol)) <= max(tol*norm(LHS), atol).
            atol: The absolute tolerance for convergence,
                norm(RHS.dot(sol)) <= max(tol*norm(LHS), atol).
            x0: The initial guess for the solution.
                M{sparse matrix, dense matrix, LinearOperator}.
                If None, solvers usually start from the null vector.
            inner_m int: The number of inner GMRES iterations per outer iteration.
            outer_k: The number of vectors to carry between inner GMRES iterations.
            outer_v:  The data used to augment the Krylov subspace.
            store_outer_A: Whether LGMRES should store also A*v in
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
            preconditioner.shape != self.problem.lhs.shape
        ):
            msg = "Inconsistent Preconditioner shape: %s != %s"
            raise ValueError(msg.format(preconditioner.shape, self.problem.lhs.shape))

        if x0 is not None and len(x0) != len(self.problem.rhs):
            msg = "Inconsistent initial guess shape: %s != %s"
            raise ValueError(msg.format(x0.shape, self.problem.rhs.shape))

        if use_ilu_precond and preconditioner is not None:
            raise ValueError(
                "Use either 'use_ilu_precond' or provide 'preconditioner',"
                " but not both."
            )

        return self._process_options(
            max_iter=max_iter,
            preconditioner=preconditioner,
            tol=tol,
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
        self, **options  # type: Union[None, bool, int, float, ndarray]
    ):  # type: (...) -> ndarray
        """Run the algorithm.

        Args:
            **options: The options for the algorithm.

        Returns:
            The solution of the problem.
        """
        if issparse(self.problem.rhs):
            self.problem.rhs = self.problem.rhs.toarray()
        rhs = self.problem.rhs
        lhs = self.problem.lhs

        opts_solver = options.copy()
        c_dtype = None

        if rhs.dtype != lhs.dtype and not isinstance(lhs, LinearOperator):
            c_dtype = find_common_type([rhs.dtype, lhs.dtype], [])
            if lhs.dtype != c_dtype:
                lhs = lhs.astype(c_dtype)
            if rhs.dtype != c_dtype:
                rhs = rhs.astype(c_dtype)

        if opts_solver["use_ilu_precond"] and not isinstance(lhs, LinearOperator):
            opts_solver["M"] = self._build_ilu_preconditioner(lhs, c_dtype)

        del opts_solver["use_ilu_precond"]
        del opts_solver["store_residuals"]

        method = self.methods_map[self.algo_name]

        if options.get("store_residuals", False):
            opts_solver["callback"] = self.__store_residuals

        opts_solver.pop("save_when_fail")
        self.problem.solution, info = method(lhs, rhs, **opts_solver)
        self._check_solver_info(info, options)

        return self.problem.solution

    def __store_residuals(
        self, current_x  # type: ndarray
    ):  # type: (...) -> ndarray
        """Store the current iteration residuals.

        Args:
            current_x: The current solution.
        """
        self.problem.solution = current_x
        self.problem.compute_residuals(True, True)

    def _check_solver_info(
        self,
        info,  # type: int
        options,  # type: Mapping[str, Any]
    ):  # type: (...) -> bool
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
        self.problem.is_converged = info == 0

        if info > 0:
            if self.problem.solution is not None:
                LOGGER.warning(
                    "%s, residual = %s",
                    self.BASE_INFO_MSG,
                    self.problem.compute_residuals(True),
                )
                LOGGER.warning("info = %s", info)
            else:
                LOGGER.warning("Failed to converge")
            return False

        # check the dimensions
        if info < 0:
            raise RuntimeError(
                self.BASE_INFO_MSG + "illegal input or breakdown" ", options = %s",
                options,
            )

        return True

    def _run_default_solver(
        self,
        lhs,  # type: Union[ndarray, spmatrix, LinearOperator]
        rhs,  # type: ndarray
        **options  # type: Any
    ):  # type: (...) -> Tuple[ndarray, int]
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
        min_res = self.problem.compute_residuals(True, current_x=best_sol)

        if self._check_solver_info(info, options):
            # If converged, stop
            return best_sol, info
        else:
            # Otherwise try GMRES
            min_res = self.problem.compute_residuals(True, current_x=best_sol)
            for k in self.LGMRES_SPEC_OPTS:  # Adapt options
                if k in options:
                    del options[k]

            if min_res < 1.0:
                options["x0"] = best_sol

            sol, info = gmres(A=lhs, b=rhs, **options)
            res = self.problem.compute_residuals(True, current_x=sol)

            if res < min_res:
                best_sol = sol
                options["x0"] = best_sol

            if self._check_solver_info(info, options):  # pragma: no cover
                return best_sol, info

        # In this case previous runs failed, trying direct method
        # based on super LU
        a_fact = splu(lhs)
        sol = a_fact.solve(rhs)
        res = self.problem.compute_residuals(True, current_x=best_sol)

        if res < options["tol"]:  # pragma: no cover
            best_sol = sol
            info = 0
            self.problem.is_converged = True
        else:
            info = 1

        return best_sol, info
