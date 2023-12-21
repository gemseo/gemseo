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
"""Tests for the module main_discipline."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from numpy import array
from numpy import zeros
from numpy.testing import assert_equal

from gemseo.problems.scalable.parametric.core.disciplines.main_discipline import (
    MainDiscipline,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


@pytest.fixture(scope="module")
def default_input_values() -> dict[str, NDArray[float]]:
    """The default input values."""
    return {
        "x_0": array([0.0, 0.0, 0.0, 0.0]),
        "y_1": array([0.0, 0.0, 0.0]),
        "y_2": array([0.0, 0.0]),
    }


@pytest.fixture(scope="module")
def main_discipline(default_input_values) -> MainDiscipline:
    """The main discipline."""
    return MainDiscipline(
        array([1.0, 2.0, 4.0]), array([-1.0, -2.0]), **default_input_values
    )


def test_name(main_discipline):
    """Check the name of the main discipline."""
    assert main_discipline.name == "MainDiscipline"


def test_input_names(main_discipline):
    """Check the input names of the main discipline."""
    assert main_discipline.input_names == ["x_0", "y_1", "y_2"]


def test_output_names(main_discipline):
    """Check the output names of the main discipline."""
    assert main_discipline.output_names == ["f", "c_1", "c_2"]


def test_input_names_to_default_values(main_discipline, default_input_values):
    """Check the default values of the input variables."""
    assert_equal(main_discipline.input_names_to_default_values, default_input_values)


def test_names_to_sizes(main_discipline, default_input_values):
    """Check the sizes of the variables."""
    assert main_discipline.names_to_sizes == {
        "c_1": 3,
        "c_2": 2,
        "f": 1,
        "x_0": 4,
        "y_1": 3,
        "y_2": 2,
    }


def test_execute_default(main_discipline):
    """Check the execution of the discipline with default values."""
    assert_equal(
        main_discipline(),
        {
            "f": array([0.0]),
            "c_1": array([1.0, 2.0, 4.0]),
            "c_2": array([-1.0, -2.0]),
        },
    )


def test_execute_x_0(main_discipline):
    """Check the execution of the discipline with custom x_0."""
    assert_equal(
        main_discipline(x_0=array([1.0, 2.0, 1.0, 2.0])),
        {
            "f": array([10.0]),
            "c_1": array([1.0, 2.0, 4.0]),
            "c_2": array([-1.0, -2.0]),
        },
    )


def test_execute_y_2(main_discipline):
    """Check the execution of the discipline with custom y_2."""
    assert_equal(
        main_discipline(y_2=array([1.0, 2.0])),
        {
            "f": array([5.0]),
            "c_1": array([1.0, 2.0, 4.0]),
            "c_2": array([-2.0, -4.0]),
        },
    )


def test_differentiation_default(main_discipline):
    """Check the differentiation of the discipline with default values."""
    assert_equal(
        main_discipline(compute_jacobian=True),
        {
            "f": {
                "x_0": array([[0.0, 0.0, 0.0, 0.0]]),
                "y_1": array([[0.0, 0.0, 0.0]]),
                "y_2": array([[0.0, 0.0]]),
            },
            "c_1": {
                "x_0": zeros((3, 4)),
                "y_1": array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.00]]),
                "y_2": zeros((3, 2)),
            },
            "c_2": {
                "x_0": zeros((2, 4)),
                "y_1": zeros((2, 3)),
                "y_2": array([[-1.0, 0.0], [0.0, -1.0]]),
            },
        },
    )


def test_differentiation_x_0(main_discipline):
    """Check the differentiation of the discipline with custom x_0."""
    assert_equal(
        main_discipline(x_0=array([1.0, 2.0, 1.0, 2.0]), compute_jacobian=True),
        {
            "f": {
                "x_0": array([[2.0, 4.0, 2.0, 4.0]]),
                "y_1": array([[0.0, 0.0, 0.0]]),
                "y_2": array([[0.0, 0.0]]),
            },
            "c_1": {
                "x_0": zeros((3, 4)),
                "y_1": array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]),
                "y_2": zeros((3, 2)),
            },
            "c_2": {
                "x_0": zeros((2, 4)),
                "y_1": zeros((2, 3)),
                "y_2": array([[-1.0, 0.0], [0.0, -1.0]]),
            },
        },
    )


def test_differentiation_y_2(main_discipline):
    """Check the differentiation of the discipline with custom y_2."""
    assert_equal(
        main_discipline(y_2=array([1.0, 2.0]), compute_jacobian=True),
        {
            "f": {
                "x_0": array([[0.0, 0.0, 0.0, 0.0]]),
                "y_1": array([[0.0, 0.0, 0.0]]),
                "y_2": array([[2.0, 4.0]]),
            },
            "c_1": {
                "x_0": zeros((3, 4)),
                "y_1": array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]),
                "y_2": zeros((3, 2)),
            },
            "c_2": {
                "x_0": zeros((2, 4)),
                "y_1": zeros((2, 3)),
                "y_2": array([[-1.0, 0.0], [0.0, -1.0]]),
            },
        },
    )
