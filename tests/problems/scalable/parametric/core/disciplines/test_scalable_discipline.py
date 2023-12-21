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
"""Tests for the module scalable_discipline."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from numpy import array
from numpy.testing import assert_equal

from gemseo.problems.scalable.parametric.core.disciplines.scalable_discipline import (
    Coefficients,
)
from gemseo.problems.scalable.parametric.core.disciplines.scalable_discipline import (
    ScalableDiscipline,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


@pytest.fixture(scope="module")
def default_input_values() -> dict[str, NDArray[float]]:
    """The default input values."""
    return {
        "x_0": array([0.0, 0.0, 0.0]),
        "x_1": array([0.0, 0.0]),
        "y_2": array([0.0, 0.0, 0.0]),
    }


@pytest.fixture(scope="module")
def coefficients() -> Coefficients:
    """The coefficients of a scalable discipline."""
    return Coefficients(
        array([1.0, 2.0]),
        array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        array([[-1.0, 0.0], [0.0, -1.0]]),
        {"y_2": array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])},
    )


@pytest.fixture(scope="module")
def scalable_discipline(coefficients, default_input_values) -> ScalableDiscipline:
    """The scalable discipline."""
    return ScalableDiscipline(
        1,
        coefficients.a_i,
        coefficients.D_i0,
        coefficients.D_ii,
        coefficients.C_ij,
        **default_input_values,
    )


def test_name(scalable_discipline):
    """Check the name of the scalable discipline."""
    assert scalable_discipline.name == "ScalableDiscipline[1]"


def test_input_names(scalable_discipline):
    """Check the input names of the scalable discipline."""
    assert scalable_discipline.input_names == ["x_0", "x_1", "y_2"]


def test_output_names(scalable_discipline):
    """Check the output names of the scalable discipline."""
    assert scalable_discipline.output_names == ["y_1"]


def test_input_names_to_default_values(scalable_discipline, default_input_values):
    """Check the default values of the input variables."""
    assert_equal(
        scalable_discipline.input_names_to_default_values, default_input_values
    )


def test_names_to_sizes(scalable_discipline):
    """Check the sizes of the variables."""
    assert scalable_discipline.names_to_sizes == {
        "x_1": 2,
        "x_0": 3,
        "y_1": 2,
        "y_2": 3,
    }


def test_coefficients(scalable_discipline, coefficients):
    """Check the coefficients of the scalable discipline."""
    assert_equal(scalable_discipline.coefficients.D_i0, coefficients.D_i0)
    assert_equal(scalable_discipline.coefficients.D_ii, coefficients.D_ii)
    assert_equal(scalable_discipline.coefficients.C_ij, coefficients.C_ij)
    assert_equal(scalable_discipline.coefficients.a_i, coefficients.a_i)


def test_execute_default(scalable_discipline):
    """Check the execution of the discipline with default values."""
    assert_equal(scalable_discipline(), {"y_1": array([1.0, 2.0])})


def test_execute_x_0(scalable_discipline):
    """Check the execution of the discipline with custom x_0."""
    assert_equal(
        scalable_discipline(x_0=array([1.0, 2.0, 1.0])), {"y_1": array([0.0, 0.0])}
    )


def test_execute_x_i(scalable_discipline):
    """Check the execution of the discipline with custom x_i."""
    assert_equal(scalable_discipline(x_i=array([1.0, 2.0])), {"y_1": array([2.0, 4.0])})


def test_execute_y_1(scalable_discipline):
    """Check the execution of the discipline with custom coupling."""
    assert_equal(
        scalable_discipline(y_2=array([1.0, 2.0, 3.0])),
        {"y_1": array([15.0, -12.0])},
    )


def test_differentiation_default(scalable_discipline):
    """Check the differentiation of the discipline with default values."""
    assert_equal(
        scalable_discipline(compute_jacobian=True),
        {
            "y_1": {
                "x_1": array([[1.0, 0.0], [0.0, 1.0]]),
                "x_0": array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]),
                "y_2": array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]),
                "u_1": array([[1.0, 0.0], [0.0, 1.0]]),
            }
        },
    )


def test_differentiation_x_0(scalable_discipline):
    """Check the differentiation of the discipline with custom x_0."""
    assert_equal(
        scalable_discipline(x_0=array([1.0, 2.0, 1.0]), compute_jacobian=True),
        {
            "y_1": {
                "x_1": array([[1.0, 0.0], [0.0, 1.0]]),
                "x_0": array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]),
                "y_2": array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]),
                "u_1": array([[1.0, 0.0], [0.0, 1.0]]),
            }
        },
    )


def test_differentiation_y_2(scalable_discipline):
    """Check the differentiation of the discipline with custom y_2."""
    assert_equal(
        scalable_discipline(y_2=array([1.0, 2.0, 3.0]), compute_jacobian=True),
        {
            "y_1": {
                "x_1": array([[1.0, -0.0], [-0.0, 1.0]]),
                "x_0": array([[-1.0, -0.0, -0.0], [-0.0, -1.0, -0.0]]),
                "y_2": array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]),
                "u_1": array([[1.0, 0.0], [0.0, 1.0]]),
            }
        },
    )
