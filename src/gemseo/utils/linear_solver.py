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
#        :author: Francois Gallard, Charlie Vanaret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Linear solvers wrapper
**********************
"""
from __future__ import division, unicode_literals

import logging

import numpy as np
import scipy.sparse.linalg as scipy_linalg
from scipy.sparse.base import issparse
from scipy.sparse.linalg import bicgstab, cgs

LOGGER = logging.getLogger(__name__)


class LinearSolver(object):
    """Solve a linear system Ax=b."""

    LGMRES = "lgmres"
    AVAILABLE_SOLVERS = {LGMRES: scipy_linalg.lgmres}

    def __init__(self):
        """Constructor."""
        self.outer_v = []  # Used to store (v,Av) pairs for restart and multiple RHS

    @staticmethod
    def _check_linear_solver(linear_solver):
        """Check that linear solver is available.

        :param linear_solver: name of linear solver used to solve linear solver
        """
        solver = LinearSolver.AVAILABLE_SOLVERS.get(linear_solver, None)
        if solver is None:
            raise AttributeError(
                "Invalid linear solver" + str(linear_solver) + " for scipy sparse: "
            )
        return solver

    @staticmethod
    def _check_b(a_mat, b_vec):
        """Check the dimensions of the vector b converts it to ndarray if sparse, for
        lgmres needs.

        :param a_mat: the matrix A
        :param b_vec: the vector b
        :returns: the vector b with consistent dimensions
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

    def solve(self, a_mat, b_vec, linear_solver="lgmres", **kwargs_lin):
        """Solves the linear system Ax=b using scipy sparse GMRES solver.

        :param a_mat: matrix A of the system, can be a sparse matrix
        :param b_vec: second member
        :param linear_solver: name of linear solver (Default value = 'lgmres')
        :param kwargs_lin: arguments passed to the scipy linear solver
        :returns: solution x such that A.x=b
        """
        scipy_linear_solver = LinearSolver._check_linear_solver(linear_solver)

        # check the dimensions of b
        b_vec = LinearSolver._check_b(a_mat, b_vec)
        # solve the system
        if "tol" not in kwargs_lin:
            kwargs_lin["tol"] = 1e-8
        kwargs_lin["atol"] = kwargs_lin["tol"]

        if "maxiter" not in kwargs_lin:
            kwargs_lin["maxiter"] = 50 * len(b_vec)
        else:
            kwargs_lin["maxiter"] = min(kwargs_lin["maxiter"], 50 * len(b_vec))
        sol, info = scipy_linear_solver(
            A=a_mat, b=b_vec, outer_v=self.outer_v, **kwargs_lin
        )
        base_msg = "scipy linear solver algorithm stop info: "
        if info > 0:
            msg = "convergence to tolerance not achieved, number of iterations"
            total_msg = base_msg + msg
            LOGGER.warning(total_msg)
            total_msg = base_msg + "--- trying bicgstab method"
            LOGGER.warning(total_msg)

            sol, info = bicgstab(
                a_mat, b_vec, sol, maxiter=50 * len(b_vec), atol=kwargs_lin["atol"]
            )
            diff = a_mat.dot(sol) - b_vec.T
            res = np.sqrt(np.sum(diff))

            total_msg = "{} --- --- residual = {}".format(base_msg, res)
            LOGGER.warning(total_msg)
            total_msg = "{} --- --- info = {}".format(base_msg, info)
            LOGGER.warning(total_msg)

            if info < 0:
                total_msg = "{} --- trying cgs method".format(base_msg)
                LOGGER.warning(total_msg)

                sol, info = cgs(
                    a_mat, b_vec, sol, maxiter=50 * len(b_vec), atol=kwargs_lin["atol"]
                )
                diff = a_mat.dot(sol) - b_vec.T
                res = np.sqrt(np.sum(diff))

                total_msg = "{} --- --- residual = {}".format(base_msg, res)
                LOGGER.warning(total_msg)
                total_msg = "{} --- --- info = {}".format(base_msg, info)
                LOGGER.warning(total_msg)
        elif info < 0:
            msg = "illegal input or breakdown"
            total_msg = base_msg + msg
            LOGGER.error(total_msg)
        return np.atleast_2d(sol).T
