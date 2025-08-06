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

from __future__ import annotations

import re

import pytest
from numpy import array
from numpy.testing import assert_equal

from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.disciplines.taylor import TaylorDiscipline


@pytest.fixture(
    params=[
        {},
        {"alpha": array([1.0]), "beta": array([0.0])},
    ]
)
def input_data(request):
    return request.param


@pytest.fixture
def discipline(linear_combination, input_data) -> TaylorDiscipline:
    """A Taylor discipline."""
    return TaylorDiscipline(linear_combination, input_data=input_data)


def test_default_outputs(discipline):
    """Test the default_outputs.

    This is the first term of the Taylor polynomial.
    """
    delta = -1.0 if discipline.io.input_grammar.defaults["alpha"] == 1.0 else -2.0
    assert discipline.io.output_grammar.defaults == {"delta": array([delta])}


def test_io_names(linear_combination, discipline) -> None:
    """Test the input and output names."""
    assert set(linear_combination.io.input_grammar) == set(discipline.io.input_grammar)
    assert set(linear_combination.io.output_grammar) == set(
        discipline.io.output_grammar
    )


def test_execute(linear_combination, discipline, input_data) -> None:
    """Test the execute() method."""
    assert_equal(discipline.execute(input_data), linear_combination.execute(input_data))


def test_linearize(linear_combination, discipline) -> None:
    """Test the linearize() method."""
    assert_equal(
        discipline.linearize(compute_all_jacobians=True),
        linear_combination.linearize(compute_all_jacobians=True),
    )


@pytest.mark.parametrize(
    ("input_data", "default_input_data"),
    [
        ({}, {"alpha": array([0.0])}),
        (
            {"alpha": array([1.0])},
            {"alpha": array([0.0]), "beta": array([0.0])},
        ),
    ],
)
def test_raises_wrong_instantiation(linear_combination, input_data, default_input_data):
    """Tests that TaylorDiscipline requires either input data or default inputs."""
    linear_combination.io.input_grammar.defaults = default_input_data
    with pytest.raises(
        ValueError,
        match=re.escape(
            "All the discipline input values must be specified either in input_data or "
            "in discipline.io.input_grammar.defaults."
        ),
    ):
        return TaylorDiscipline(linear_combination, input_data=input_data)


@pytest.fixture(scope="module")
def analytic_discipline() -> AnalyticDiscipline:
    """An analytic discipline."""
    return AnalyticDiscipline({"y1": "1+2*x1+3*x2", "y2": "1-2*x1-3*x2"})


@pytest.mark.parametrize(
    ("input_data", "expected"),
    [({}, (1.0, -5.0)), ({"x1": array([1.0]), "x2": array([2.0])}, (-7.0, -11.0))],
)
def test_input_output_names(analytic_discipline, input_data, expected):
    """Test that TaylorDiscipline can be restricted to some IO variables."""
    taylor = TaylorDiscipline(
        analytic_discipline,
        input_names=["x1"],
        output_names=["y2"],
        input_data=input_data,
    )
    assert taylor.io.input_grammar.keys() == {"x1"}
    assert taylor.io.output_grammar.keys() == {"y2"}

    taylor.execute()
    assert taylor.io.data["y2"] == expected[0]

    taylor.execute({"x1": array([3.0])})
    assert taylor.io.data["y2"] == expected[1]
