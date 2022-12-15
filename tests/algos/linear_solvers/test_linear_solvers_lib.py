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
from __future__ import annotations

import pytest
from gemseo.algos._unsuitability_reason import _UnsuitabilityReason
from gemseo.algos.linear_solvers.linear_problem import LinearProblem
from gemseo.algos.linear_solvers.linear_solver_lib import LinearSolverDescription
from gemseo.algos.linear_solvers.linear_solver_lib import LinearSolverLib
from numpy import eye
from numpy import ones
from scipy.sparse.linalg import aslinearoperator
from scipy.sparse.linalg import LinearOperator


@pytest.mark.parametrize("is_symmetric", [False, True])
@pytest.mark.parametrize("lhs_must_be_symmetric", [False, True])
def test_linear_solver_for_symmetric_lhs(is_symmetric, lhs_must_be_symmetric):
    """Check is_algorithm_suited with linear solver requiring symmetric LHS."""
    description = LinearSolverDescription(
        algorithm_name="foo",
        internal_algorithm_name="bar",
        lhs_must_be_symmetric=lhs_must_be_symmetric,
    )
    problem = LinearProblem(eye(2), ones(2), is_symmetric=is_symmetric)
    is_suited = LinearSolverLib.is_algorithm_suited(description, problem)
    assert is_suited is (not lhs_must_be_symmetric or is_symmetric)
    if not is_suited:
        assert (
            LinearSolverLib._get_unsuitability_reason(description, problem)
            == _UnsuitabilityReason.NOT_SYMMETRIC
        )


@pytest.mark.parametrize("is_positive_def", [False, True])
@pytest.mark.parametrize("lhs_must_be_positive_definite", [False, True])
def test_linear_solver_for_positive_definite_lhs(
    is_positive_def, lhs_must_be_positive_definite
):
    """Check is_algorithm_suited with linear solver requiring positive definite LHS."""
    description = LinearSolverDescription(
        algorithm_name="foo",
        internal_algorithm_name="bar",
        lhs_must_be_positive_definite=lhs_must_be_positive_definite,
    )
    problem = LinearProblem(eye(2), ones(2), is_positive_def=is_positive_def)
    is_suited = LinearSolverLib.is_algorithm_suited(description, problem)
    assert is_suited is (not lhs_must_be_positive_definite or is_positive_def)
    if not is_suited:
        assert (
            LinearSolverLib._get_unsuitability_reason(description, problem)
            == _UnsuitabilityReason.NOT_POSITIVE_DEFINITE
        )


@pytest.mark.parametrize("lhs_must_be_linear_operator", [False, True])
@pytest.mark.parametrize("lhs", [aslinearoperator(eye(2)), eye(2)])
def test_linear_solver_for_linear_operator(lhs, lhs_must_be_linear_operator):
    """Check is_algorithm_suited with linear solver requiring linear operator."""
    description = LinearSolverDescription(
        algorithm_name="foo",
        internal_algorithm_name="bar",
        lhs_must_be_linear_operator=lhs_must_be_linear_operator,
    )
    problem = LinearProblem(lhs, ones(2))
    is_suited = LinearSolverLib.is_algorithm_suited(description, problem)
    assert is_suited is (
        lhs_must_be_linear_operator or not isinstance(lhs, LinearOperator)
    )
    if not is_suited:
        assert (
            LinearSolverLib._get_unsuitability_reason(description, problem)
            == _UnsuitabilityReason.NOT_LINEAR_OPERATOR
        )
