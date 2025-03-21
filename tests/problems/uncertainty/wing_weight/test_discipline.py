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

import pytest
from numpy import array
from numpy import cos
from numpy import ndarray
from numpy import pi

from gemseo.problems.uncertainty.wing_weight.discipline import WingWeightDiscipline


def compute_output(input_values):
    a, lamda, nz, sw, wdg, wfw, wp, ell, q, tc = input_values
    return (
        0.036
        * sw**0.758
        * wfw**0.0035
        * (a / cos(pi / 180 * lamda) ** 2) ** 0.6
        * q**0.006
        * ell**0.04
        * (100 * tc / cos(pi / 180 * lamda)) ** (-0.3)
        * (nz * wdg) ** 0.49
        + sw * wp
    )


@pytest.fixture(scope="module")
def input_data_as_dict() -> dict[str, ndarray]:
    """The input values of interest as a dictionary."""
    return {
        name: array([1.0])
        for name in ["A", "Lamda", "Nz", "Sw", "Wdg", "Wfw", "Wp", "ell", "q", "tc"]
    }


@pytest.fixture(scope="module")
def input_data_as_array() -> ndarray:
    """The input values of interest as an array."""
    return array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])


@pytest.fixture(scope="module")
def discipline() -> WingWeightDiscipline:
    """The discipline of the wing weight problem."""
    return WingWeightDiscipline()


def test_init(discipline) -> None:
    """Check the instantiation of the discipline."""
    input_names = ["A", "Lamda", "Nz", "Sw", "Wdg", "Wfw", "Wp", "ell", "q", "tc"]
    default_values = [8, 0.0, 4.25, 175, 2100, 260, 0.0525, 0.75, 14.5, 0.13]
    assert discipline.name == "WingWeightDiscipline"
    assert list(discipline.io.input_grammar) == input_names
    assert list(discipline.io.output_grammar) == ["Ww"]
    assert discipline.io.input_grammar.defaults == {
        name: array([default_values[i]]) for i, name in enumerate(input_names)
    }


def test_execute(discipline, input_data_as_dict, input_data_as_array) -> None:
    """Check the output value of the discipline."""
    assert discipline.execute(input_data_as_dict)["Ww"][0] == compute_output(
        input_data_as_array
    )
