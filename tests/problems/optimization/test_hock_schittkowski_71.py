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

"""Tests for the Hock and Schittkowski problem 71."""

from __future__ import annotations

import pytest
from numpy import array

from gemseo.problems.optimization.hock_schittkowski_71 import HockSchittkowski71


@pytest.fixture
def hock_schittkowski_71() -> HockSchittkowski71:
    """Create a :class:`.HockSchittkowski71` optimization problem.

    Returns:
        A HockSchittkowski71 instance.
    """
    return HockSchittkowski71(initial_guess=array([1.0, 5.0, 5.0, 1.0]))


def test_obj_jacobian(hock_schittkowski_71):
    """Test the Jacobian of the Hock and Schittkowski 71 objective function.

    Args:
        hock_schittkowski_71: Fixture returning a HockSchittkowski71
            `OptimizationProblem`.
    """
    x_dv = array([1.0, 1.0, 1.0, 1.0])
    hock_schittkowski_71.objective.check_grad(x_dv, error_max=1e-6)


def test_equality_constraint_jacobian(hock_schittkowski_71):
    """Test the equality constraint function's Jacobian  of Hock and Schittkowski 71.

    Args:
        hock_schittkowski_71: Fixture returning a HockSchittkowski71
            `OptimizationProblem`.
    """
    x_dv = array([1.0, 1.0, 1.0, 1.0])
    hock_schittkowski_71.constraints[0].check_grad(x_dv, error_max=1e-6)


def test_inequality_constraint_jacobian(hock_schittkowski_71):
    """Test the inequality constraint function's Jacobian  of Hock and Schittkowski 71.

    Args:
        hock_schittkowski_71: Fixture returning a HockSchittkowski71
            `OptimizationProblem`.
    """
    x_dv = array([1.0, 1.0, 1.0, 1.0])
    hock_schittkowski_71.constraints[1].check_grad(x_dv, error_max=1e-6)


def test_compute_objective(hock_schittkowski_71):
    """Test the objective function of the Hock and Schittkowski 71 problem.

    Args:
        hock_schittkowski_71: Fixture returning a HockSchittkowski71
        `OptimizationProblem`.
    """
    x_dv = array([1.0, 1.0, 1.0, 1.0])
    objective = hock_schittkowski_71.compute_objective(x_dv)

    assert objective == 4.0


def test_compute_equality_constraint(hock_schittkowski_71):
    """Test the equality constraint function of the Hock and Schittkowski 71 problem.

    Args:
        hock_schittkowski_71: Fixture returning a HockSchittkowski71
        `OptimizationProblem`.
    """
    x_dv = array([1.0, 1.0, 1.0, 1.0])
    equality_constraint = hock_schittkowski_71.compute_equality_constraint(x_dv)
    assert equality_constraint == -36.0


def test_compute_inequality_constraint(hock_schittkowski_71):
    """Test the inequality constraint function of the Hock and Schittkowski 71 problem.

    Args:
        hock_schittkowski_71: Fixture returning a HockSchittkowski71
        `OptimizationProblem`.
    """
    x_dv = array([1.0, 1.0, 1.0, 1.0])
    inequality_constraint = hock_schittkowski_71.compute_inequality_constraint(x_dv)
    assert inequality_constraint == 24.0


def test_solution(hock_schittkowski_71) -> None:
    """Check the objective value at the solution.

    Args:
      hock_schittkowski_71: Fixture returning a HockSchittkowski71
      `OptimizationProblem`.
    """
    x_opt, f_opt = hock_schittkowski_71.get_solution()
    assert hock_schittkowski_71.objective.evaluate(x_opt) == pytest.approx(f_opt)
