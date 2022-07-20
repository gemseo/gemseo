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
#        :author: Francois Gallard, Charlie Vanaret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Linear solvers wrapper
**********************
"""
from __future__ import annotations

import logging

import numpy as np
import scipy.sparse.linalg as scipy_linalg
from scipy.sparse.base import issparse
from scipy.sparse.linalg import bicgstab
from scipy.sparse.linalg import cgs

LOGGER = logging.getLogger(__name__)


class LinearSolver:
    """Solve a linear system Ax=b."""

    LGMRES = "lgmres"
    AVAILABLE_SOLVERS = {LGMRES: scipy_linalg.lgmres}

    def __init__(self):
        """Constructor."""
        self.outer_v = []  # Used to store (v,Av) pairs for restart and multiple RHS

    @staticmethod
    def _check_linear_solver(linear_solver):
        """Check that linear solver is available.

        Args:
            linear_solver: The name of the linear solver to solve the linear problem.
        """
        solver = LinearSolver.AVAILABLE_SOLVERS.get(linear_solver, None)
        if solver is None:
            raise AttributeError(
                "Invalid linear solver" + str(linear_solver) + " for scipy sparse: "
            )
        return solver

    @staticmethod
    def _check_b(a_mat, b_vec):
        """Check the dimensions of the vector b and convert it to ndarray if sparse.

        For lgmres needs.

        Args:
            a_mat: The matrix A.
            b_vec: The vector b.

        Returns:
            The vector b with consistent dimensions.
        """
        if len(b_vec.shape) == 2 and b_vec.shape[1] != 1:
            LOGGER.error(
                "Incompatible dimensions in linear system Ax=b, A "
                "shape is %s and b shape is %s",
                str(a_mat.shape),
                str(b_vec.shape),
            )
            raise ValueError(
                "Second member of the linear system" + " must be a column vector"
            )
        if issparse(b_vec):
            b_vec = b_vec.toarray()
        return b_vec.real

    def solve(self, a_mat, b_vec, linear_solver="lgmres", **options):
        """Solve the linear system :math:`Ax=b`.

        Args:
            a_mat: The matrix :math:`A` of the system, can be a sparse matrix.
            b_vec: The second member :math:`b` of the system.
            linear_solver: The name of linear solver.
            **options: The options of the linear solver.

        Returns:
            The solution :math:`x` such that :math:`Ax=b`.
        """
        scipy_linear_solver = LinearSolver._check_linear_solver(linear_solver)

        # check the dimensions of b
        b_vec = LinearSolver._check_b(a_mat, b_vec)
        # solve the system
        if "tol" not in options:
            options["tol"] = 1e-8
        options["atol"] = options["tol"]

        if "maxiter" not in options:
            options["maxiter"] = 50 * len(b_vec)
        else:
            options["maxiter"] = min(options["maxiter"], 50 * len(b_vec))
        sol, info = scipy_linear_solver(
            A=a_mat, b=b_vec, outer_v=self.outer_v, **options
        )
        base_msg = "scipy linear solver algorithm stop info: "
        if info > 0:
            msg = "convergence to tolerance not achieved, number of iterations"
            total_msg = base_msg + msg
            LOGGER.warning(total_msg)
            total_msg = base_msg + "--- trying bicgstab method"
            LOGGER.warning(total_msg)

            sol, info = bicgstab(
                a_mat, b_vec, sol, maxiter=50 * len(b_vec), atol=options["atol"]
            )
            diff = a_mat.dot(sol) - b_vec.T
            res = np.sqrt(np.sum(diff))

            total_msg = f"{base_msg} --- --- residual = {res}"
            LOGGER.warning(total_msg)
            total_msg = f"{base_msg} --- --- info = {info}"
            LOGGER.warning(total_msg)

            if info < 0:
                total_msg = f"{base_msg} --- trying cgs method"
                LOGGER.warning(total_msg)

                sol, info = cgs(
                    a_mat, b_vec, sol, maxiter=50 * len(b_vec), atol=options["atol"]
                )
                diff = a_mat.dot(sol) - b_vec.T
                res = np.sqrt(np.sum(diff))

                total_msg = f"{base_msg} --- --- residual = {res}"
                LOGGER.warning(total_msg)
                total_msg = f"{base_msg} --- --- info = {info}"
                LOGGER.warning(total_msg)
        elif info < 0:
            msg = "illegal input or breakdown"
            total_msg = base_msg + msg
            LOGGER.error(total_msg)
        return np.atleast_2d(sol).T
