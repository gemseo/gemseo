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

import operator

import pytest
from numpy import arange
from numpy import array
from numpy import eye
from numpy import ndarray
from numpy.testing import assert_equal

from gemseo.discipline.array_based_function import ArrayBasedFunctionDiscipline
from gemseo.util.testing.helper import assert_exception


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
    assert_equal(discipline.io.input_data["x1"], array([0.0]))
    assert_equal(discipline.io.input_data["x2"], array([0.0, 0.0, 0.0, 0.0]))
    assert_equal(discipline.io.output_data["y1"], array([0.0, 0.0, 0.0]))
    assert_equal(discipline.io.output_data["y2"], array([0.0, 0.0]))


def test_custom_input_values(discipline):
    """Check the ArrayBasedFunctionDiscipline with custom input values."""
    discipline.execute({"x1": array([1.0]), "x2": array([2.0, 3.0, 4.0, 5.0])})
    assert_equal(discipline.io.input_data["x1"], array([1.0]))
    assert_equal(discipline.io.input_data["x2"], array([2.0, 3.0, 4.0, 5.0]))
    assert_equal(discipline.io.output_data["y1"], array([1.0, 2.0, 3.0]))
    assert_equal(discipline.io.output_data["y2"], array([4.0, 5.0]))


def test_input_order():
    """Check that the inputs are concatenated in the order of the instantiation.

    The default input values are set in another order
    and passed as is to _run when executing the discipline without input data.
    """
    discipline = ArrayBasedFunctionDiscipline(
        lambda x: eye(5) @ x, {"x1": 1, "x2": 4}, {"y1": 3, "y2": 2}
    )
    discipline.io.input_grammar.defaults = {
        "x2": array([2.0, 3.0, 4.0, 5.0]),
        "x1": array([1.0]),
    }
    discipline.execute()
    assert_equal(discipline.io.output_data["y1"], array([1.0, 2.0, 3.0]))
    assert_equal(discipline.io.output_data["y2"], array([4.0, 5.0]))


def test_jac_function_none(discipline, snapshot):
    """Check the Jacobian computation when jac_function is None."""
    with assert_exception(RuntimeError, snapshot):
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


@pytest.fixture
def matrix() -> ndarray:
    """The matrix defining the linear function."""
    return arange(1.0, 26.0).reshape((5, 5))


@pytest.fixture
def namespaced_discipline(matrix) -> ArrayBasedFunctionDiscipline:
    """An ArrayBasedFunctionDiscipline whose first input and output have a namespace.

    The namespaces move them to the end of their grammars.
    """
    discipline = ArrayBasedFunctionDiscipline(
        lambda x: matrix @ x,
        {"x1": 1, "x2": 4},
        {"y1": 3, "y2": 2},
        lambda x: matrix,
    )
    discipline.add_namespace_to_input("x1", "ns")
    discipline.add_namespace_to_output("y1", "ns")
    return discipline


def test_namespaces(namespaced_discipline, matrix):
    """Check that the namespaces do not change the order of the variables."""
    namespaced_discipline.execute({
        "ns:x1": array([1.0]),
        "x2": array([2.0, 3.0, 4.0, 5.0]),
    })
    output_vector = matrix @ arange(1.0, 6.0)
    assert_equal(namespaced_discipline.io.output_data["ns:y1"], output_vector[:3])
    assert_equal(namespaced_discipline.io.output_data["y2"], output_vector[3:])


def test_namespaces_jacobian(namespaced_discipline, matrix):
    """Check the Jacobian when the inputs and outputs have namespaces."""
    namespaced_discipline.linearize(compute_all_jacobians=True)
    jac = namespaced_discipline.jac
    assert_equal(jac["ns:y1"]["ns:x1"], matrix[:3, :1])
    assert_equal(jac["ns:y1"]["x2"], matrix[:3, 1:])
    assert_equal(jac["y2"]["ns:x1"], matrix[3:, :1])
    assert_equal(jac["y2"]["x2"], matrix[3:, 1:])


def test_name_both_input_and_output() -> None:
    """Check the Jacobian when a name is both an input and an output."""
    discipline = ArrayBasedFunctionDiscipline(
        operator.itemgetter(slice(2)), {"x": 4}, {"x": 2}, lambda x: eye(2, 4)
    )
    jacobian = discipline.linearize({"x": arange(4.0)}, compute_all_jacobians=True)
    assert_equal(jacobian["x"]["x"], eye(2, 4))
