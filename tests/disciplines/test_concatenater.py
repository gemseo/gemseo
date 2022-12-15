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
"""Test for the :class:`.Concatenater`"""
from __future__ import annotations

import pytest
from gemseo.api import create_discipline
from gemseo.disciplines.concatenater import Concatenater
from numpy import array
from numpy import concatenate
from numpy import diag
from numpy import ndarray
from numpy import ones
from numpy import zeros
from numpy.testing import assert_array_equal


@pytest.fixture(params=[None, {"c_1": 1.0, "c_2": -1.0}])
def coefficients(request):
    """The coefficients to scale the input variables."""
    return request.param


@pytest.fixture()
def concatenation_disc(coefficients):
    """Set-up fixture, creating a concatenation discipline."""
    return create_discipline(
        "Concatenater",
        input_variables=["c_1", "c_2"],
        output_variable="c",
        input_coefficients=coefficients,
    )


@pytest.fixture()
def input_data():
    """"""
    return {"c_1": array([2.0, 3.0]), "c_2": array([2.0, 3.0, 4.0])}


def test_concatenation_discipline_execution(
    concatenation_disc: Concatenater,
    input_data: dict[str, ndarray],
    coefficients: dict[str, float],
):
    """Check the output data returned by Concatenater."""
    if coefficients is None:
        coefficients = {"c_1": 1.0, "c_2": 1.0}
    output_data = concatenation_disc.execute(input_data)
    input_names = concatenation_disc.get_input_data_names()
    expected = concatenate(
        [
            input_data[input_name] * coefficients[input_name]
            for input_name in input_names
        ]
    )
    assert_array_equal(output_data["c"], expected)


def test_concatenation_discipline_linearization(
    concatenation_disc: Concatenater,
    input_data: dict[str, ndarray],
    coefficients: dict[str, float],
):
    """Check the Jacobian data returned by Concatenater."""
    if coefficients is None:
        coefficients = {"c_1": 1.0, "c_2": 1.0}
    jac = concatenation_disc.linearize(input_data, force_all=True)
    var_inputs = list(concatenation_disc.get_input_data_names())

    # In Python 2, we cannot assume any order in the var_inputs list
    # Then, we have to re-create the reference Jacobian matrix based on this order
    c_c1 = zeros([5, 2])
    start = 0
    for var in var_inputs:
        end = start + 2
        if var == "c_1":
            c_c1[start:end, :] = coefficients[var] * diag(ones(2))
        start += input_data[var].size
    assert_array_equal(jac["c"]["c_1"], c_c1)

    c_c2 = zeros([5, 3])
    start = 0
    for var in var_inputs:
        if var == "c_2":
            end = start + 3
            c_c2[start:end, :] = coefficients[var] * diag(ones(3))
        start += input_data[var].size
    assert_array_equal(jac["c"]["c_2"], c_c2)


def test_check_gradient(
    concatenation_disc: Concatenater, input_data: dict[str, ndarray]
):
    """Test the Jacobian computation by finite differences."""
    concatenation_disc.default_inputs = input_data
    assert concatenation_disc.check_jacobian(threshold=1e-3, step=1e-4)
