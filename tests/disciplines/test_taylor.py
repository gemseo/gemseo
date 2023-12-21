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
"""Tests for the TaylorDiscipline."""

import re

import pytest
from numpy import array
from numpy.testing import assert_equal

from gemseo.disciplines.taylor import TaylorDiscipline


@pytest.fixture(
    params=[
        {},
        {"alpha": array([1.0]), "beta": array([0.0])},
    ]
)
def input_data(request):
    return request.param


@pytest.fixture()
def discipline(linear_combination, input_data) -> TaylorDiscipline:
    """A Taylor discipline."""
    return TaylorDiscipline(linear_combination, input_data=input_data)


def test_io_names(linear_combination, discipline):
    """Test the input and output names."""
    assert set(linear_combination.get_input_data_names()) == set(
        discipline.get_input_data_names()
    )
    assert set(linear_combination.get_output_data_names()) == set(
        discipline.get_output_data_names()
    )


def test_execute(linear_combination, discipline, input_data):
    """Test the execute() method."""
    assert_equal(discipline.execute(input_data), linear_combination.execute(input_data))


def test_linearize(linear_combination, discipline):
    """Test the linearize() method."""
    assert_equal(
        discipline.linearize(compute_all_jacobians=True),
        linear_combination.linearize(compute_all_jacobians=True),
    )


@pytest.mark.parametrize(
    ("input_data", "default_inputs"),
    [
        ({}, {"alpha": array([0.0])}),
        (
            {"alpha": array([1.0])},
            {"alpha": array([0.0]), "beta": array([0.0])},
        ),
    ],
)
def test_raises_wrong_instantiation(linear_combination, input_data, default_inputs):
    """Tests that TaylorDiscipline requires either input data or default inputs."""
    linear_combination.default_inputs = default_inputs
    with pytest.raises(
        ValueError,
        match=re.escape(
            "All the discipline input values must be specified either in input_data or "
            "in discipline.default_inputs."
        ),
    ):
        return TaylorDiscipline(linear_combination, input_data=input_data)
