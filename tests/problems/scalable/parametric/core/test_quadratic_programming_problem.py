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
#                         documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Tests for the module quadratic_programming_problem."""

from __future__ import annotations

from gemseo.problems.scalable.parametric.core.quadratic_programming_problem import (
    QuadraticProgrammingProblem,
)


def test_quadratic_programming_problem_fields():
    """Check the fields of QuadraticProgrammingProblem."""
    assert QuadraticProgrammingProblem._fields == ("Q", "c", "d", "A", "b")


def test_quadratic_programming_problem_values():
    """Check that QuadraticProgrammingProblem does not modify the values."""
    problem = QuadraticProgrammingProblem(1, 2, 3, 4, 5)
    assert problem.Q == 1
    assert problem.c == 2
    assert problem.d == 3
    assert problem.A == 4
    assert problem.b == 5
