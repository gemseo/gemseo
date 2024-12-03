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
from __future__ import annotations

import re

import pytest
from numpy import array
from numpy import eye
from numpy.testing import assert_equal

from gemseo.disciplines.array_based_function import ArrayBasedFunctionDiscipline


@pytest.fixture(scope="module")
def discipline() -> ArrayBasedFunctionDiscipline:
    """An ArrayBasedFunctionDiscipline."""
    d = ArrayBasedFunctionDiscipline(
        lambda x: eye(5) @ x, {"x1": 1, "x2": 4}, {"y1": 3, "y2": 2}
    )
    d.name = "foo"
    return d


def test_input_grammar(discipline):
    """Check the input grammar."""
    assert discipline.io.input_grammar.keys() == {"x1", "x2"}


def test_output_grammar(discipline):
    """Check the output grammar."""
    assert discipline.io.output_grammar.keys() == {"y1", "y2"}


def test_default_input_values(discipline):
    """Check the ArrayBasedFunctionDiscipline with default input values."""
    discipline.execute()
    assert_equal(discipline.io.data["x1"], array([0.0]))
    assert_equal(discipline.io.data["x2"], array([0.0, 0.0, 0.0, 0.0]))
    assert_equal(discipline.io.data["y1"], array([0.0, 0.0, 0.0]))
    assert_equal(discipline.io.data["y2"], array([0.0, 0.0]))


def test_custom_input_values(discipline):
    """Check the ArrayBasedFunctionDiscipline with custom input values."""
    discipline.execute({"x1": array([1.0]), "x2": array([2.0, 3.0, 4.0, 5.0])})
    assert_equal(discipline.io.data["x1"], array([1.0]))
    assert_equal(discipline.io.data["x2"], array([2.0, 3.0, 4.0, 5.0]))
    assert_equal(discipline.io.data["y1"], array([1.0, 2.0, 3.0]))
    assert_equal(discipline.io.data["y2"], array([4.0, 5.0]))


def test_jac_function_none(discipline):
    """Check the Jacobian computation when jac_function is None."""
    with pytest.raises(
        RuntimeError,
        match=re.escape("The discipline foo cannot compute the analytic derivatives."),
    ):
        discipline.linearize(compute_all_jacobians=True)


def test_jac_function():
    """Check the Jacobian computation."""
    discipline = ArrayBasedFunctionDiscipline(
        lambda x: eye(5) @ x, {"x1": 1, "x2": 4}, {"y1": 3, "y2": 2}, lambda x: eye(5)
    )
    discipline.linearize(compute_all_jacobians=True)
    jac1 = discipline.jac["y1"]
    assert_equal(jac1["x1"], array([[1.0], [0.0], [0.0]]))
    assert_equal(
        jac1["x2"],
        array([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
    )
    jac2 = discipline.jac["y2"]
    assert_equal(jac2["x1"], array([[0.0], [0.0]]))
    assert_equal(jac2["x2"], array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]))
