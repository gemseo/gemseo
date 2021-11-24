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

from numpy import eye, ones
from scipy.sparse.linalg import aslinearoperator

from gemseo.algos.linear_solvers.linear_problem import LinearProblem
from gemseo.algos.linear_solvers.linear_solver_lib import LinearSolverLib


def test_symmetric_filter():
    lib_dict = {LinearSolverLib.LHS_MUST_BE_SYMMETRIC: True}
    problem = LinearProblem(eye(2), ones(2), is_symmetric=True)
    assert LinearSolverLib.is_algorithm_suited(lib_dict, problem)

    problem = LinearProblem(eye(2), ones(2), is_symmetric=False)
    assert not LinearSolverLib.is_algorithm_suited(lib_dict, problem)


def test_posdef_filter():
    lib_dict = {LinearSolverLib.LHS_MUST_BE_POSITIVE_DEFINITE: True}
    problem = LinearProblem(eye(2), ones(2), is_positive_def=True)
    assert LinearSolverLib.is_algorithm_suited(lib_dict, problem)

    problem = LinearProblem(eye(2), ones(2), is_positive_def=False)
    assert not LinearSolverLib.is_algorithm_suited(lib_dict, problem)


def test_linop_filter():
    lib_dict = {LinearSolverLib.LHS_CAN_BE_LINEAR_OPERATOR: True}
    problem = LinearProblem(aslinearoperator(eye(2)), ones(2))
    assert LinearSolverLib.is_algorithm_suited(lib_dict, problem)

    lib_dict = {LinearSolverLib.LHS_CAN_BE_LINEAR_OPERATOR: False}
    assert not LinearSolverLib.is_algorithm_suited(lib_dict, problem)
