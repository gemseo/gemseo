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
#        :author: Jean-Christophe Giret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Test for the Concatenater."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from numpy import array
from numpy import concatenate
from numpy import diag
from numpy import ones
from numpy import zeros
from numpy.testing import assert_array_equal

from gemseo import create_discipline
from gemseo.discipline import Concatenater

if TYPE_CHECKING:
    from numpy import ndarray
from gemseo.util.derivative.check.discipline import DisciplineJacobianChecker
from gemseo.util.testing.helper import assert_exception


@pytest.fixture(params=[None, {"c_1": 1.0, "c_2": -1.0}])
def coefficients(request):
    """The coefficients to scale the input variables."""
    return request.param


@pytest.fixture
def concatenation_disc(coefficients):
    """Set-up fixture, creating a concatenation discipline."""
    return create_discipline(
        "Concatenater",
        input_variables=["c_1", "c_2"],
        output_variable="c",
        input_coefficients=coefficients,
    )


@pytest.fixture
def input_data():
    """"""
    return {"c_1": array([2.0, 3.0]), "c_2": array([2.0, 3.0, 4.0])}


def test_concatenation_discipline_execution(
    concatenation_disc: Concatenater,
    input_data: dict[str, ndarray],
    coefficients: dict[str, float],
) -> None:
    """Check the output data returned by Concatenater."""
    if coefficients is None:
        coefficients = {"c_1": 1.0, "c_2": 1.0}
    output_data = concatenation_disc.execute(input_data)
    input_names = concatenation_disc.io.input_grammar
    expected = concatenate([
        input_data[input_name] * coefficients[input_name] for input_name in input_names
    ])
    assert_array_equal(output_data["c"], expected)


def test_concatenation_discipline_linearization(
    concatenation_disc: Concatenater,
    input_data: dict[str, ndarray],
    coefficients: dict[str, float],
) -> None:
    """Check the Jacobian data returned by Concatenater."""
    if coefficients is None:
        coefficients = {"c_1": 1.0, "c_2": 1.0}
    jac = concatenation_disc.linearize(input_data, compute_all_jacobians=True)
    var_inputs = list(concatenation_disc.io.input_grammar)

    # In Python 2, we cannot assume any order in the var_inputs list
    # Then, we have to re-create the reference Jacobian matrix based on this order
    c_c1 = zeros([5, 2])
    start = 0
    for var in var_inputs:
        end = start + 2
        if var == "c_1":
            c_c1[start:end, :] = coefficients[var] * diag(ones(2))
        start += input_data[var].size
    assert_array_equal(jac["c"]["c_1"].toarray(), c_c1)

    c_c2 = zeros([5, 3])
    start = 0
    for var in var_inputs:
        if var == "c_2":
            end = start + 3
            c_c2[start:end, :] = coefficients[var] * diag(ones(3))
        start += input_data[var].size
    assert_array_equal(jac["c"]["c_2"].toarray(), c_c2)


def test_check_gradient(
    concatenation_disc: Concatenater, input_data: dict[str, ndarray]
) -> None:
    """Test the Jacobian computation by finite differences."""
    concatenation_disc.io.input_grammar.defaults = input_data
    checker = DisciplineJacobianChecker(concatenation_disc)
    assert checker.check(atol=1e-3, rtol=1e-3, step=1e-4)


@pytest.fixture
def namespaced_concatenater() -> Concatenater:
    """A Concatenater whose first input has a namespace."""
    discipline = Concatenater(["a", "b"], "c", input_coefficients={"a": 2.0, "b": 3.0})
    discipline.add_namespace_to_input("a", "ns")
    return discipline


def test_namespaces_execution(namespaced_concatenater: Concatenater) -> None:
    """Check that the namespaces do not change the order of the concatenation."""
    output_data = namespaced_concatenater.execute({
        "ns:a": array([1.0]),
        "b": array([10.0]),
    })
    assert_array_equal(output_data["c"], array([2.0, 30.0]))


def test_namespaces_linearization(namespaced_concatenater: Concatenater) -> None:
    """Check the Jacobian of a Concatenater with a namespaced input."""
    jac = namespaced_concatenater.linearize(
        {"ns:a": array([1.0]), "b": array([10.0])}, compute_all_jacobians=True
    )
    assert set(jac["c"]) == {"ns:a", "b"}
    assert_array_equal(jac["c"]["ns:a"].toarray(), array([[2.0], [0.0]]))
    assert_array_equal(jac["c"]["b"].toarray(), array([[0.0], [3.0]]))


def test_repeated_input(snapshot) -> None:
    """Check the error when an input name is repeated."""
    with assert_exception(ValueError, snapshot):
        Concatenater(["a", "b", "a"], "c")
