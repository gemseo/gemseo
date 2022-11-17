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
from gemseo.algos.linear_solvers.linear_problem import LinearProblem
from gemseo.utils.testing import image_comparison
from numpy import diag
from numpy import eye
from numpy import ones
from numpy import random
from numpy import zeros
from scipy.sparse.linalg import aslinearoperator


def test_init():
    """Checks the linear problem initialization and residuals storage."""
    pb1 = LinearProblem(
        diag([1, 1]),
        ones(2),
        is_symmetric=True,
        is_positive_def=True,
        is_converged=False,
    )
    assert pb1.is_symmetric
    assert not pb1.is_converged
    assert pb1.is_positive_def

    assert pb1.compute_residuals(current_x=ones(2), store=True) == 0.0
    pb1.solution = ones(2)
    assert pb1.compute_residuals(store=True) == 0.0
    assert len(pb1.residuals_history) == 2


def test_residuals_checks():
    """Tests the basic checks for residuals computation."""
    pb1 = LinearProblem(diag([1, 1]))
    with pytest.raises(ValueError, match="Missing RHS"):
        pb1.compute_residuals(current_x=ones(2))

    pb1.rhs = ones(2)
    with pytest.raises(ValueError, match="Missing solution"):
        pb1.compute_residuals()


@pytest.mark.parametrize(
    "lhs,rhs",
    [(diag([1, 1]), ones(1)), (ones(1), ones(2)), (diag([1, 1]), diag([1, 1]))],
)
def test_size_checks(lhs, rhs):
    """Tests the sizes consistency checks in RHL and LHS."""
    problem = LinearProblem(lhs, rhs)
    with pytest.raises(ValueError, match="Incompatible dimensions in linear system"):
        problem.check()


@image_comparison(["residuals0"])
def test_plot_residuals(tmp_wd, pyplot_close_all):
    """Tests the residuals plot creation."""
    random.seed(1)
    n = 10
    problem = LinearProblem(random.rand(n, n), random.rand(n))

    for i in range(100):
        problem.compute_residuals(current_x=i * ones(n), store=True)
    problem.plot_residuals()


def test_plot_residuals_checks():
    """Tests the residuals plot creation."""
    problem = LinearProblem(eye(1), ones(1))
    with pytest.raises(ValueError, match="Residuals history is empty."):
        problem.plot_residuals()


def test_residuals():
    problem = LinearProblem(eye(3), ones(3))
    assert problem.compute_residuals(False, current_x=zeros(3)) == (3**0.5)
    assert problem.compute_residuals(current_x=zeros(3)) == 1.0
    assert problem.compute_residuals(current_x=ones(3)) == 0.0


def test_linear_operator():
    """Tests the sizes consistency checks in RHL and LHS."""
    problem = LinearProblem(aslinearoperator(eye(3)), ones(3))
    assert problem.compute_residuals(current_x=zeros(3)) == 1.0
    assert problem.is_lhs_linear_operator
