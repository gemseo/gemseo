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
from gemseo.uncertainty.use_cases.ishigami.functions import compute_gradient
from gemseo.uncertainty.use_cases.ishigami.functions import compute_output
from gemseo.uncertainty.use_cases.ishigami.ishigami_discipline import IshigamiDiscipline
from numpy import array
from numpy import ndarray
from numpy.testing import assert_equal


@pytest.fixture(scope="module")
def input_values() -> tuple[dict[str, ndarray], ndarray]:
    """The input values of interest."""
    return {name: array([1.0]) for name in ["x1", "x2", "x3"]}, array([1.0, 1.0, 1.0])


@pytest.fixture(scope="module")
def discipline() -> IshigamiDiscipline:
    """The Ishigami discipline."""
    return IshigamiDiscipline()


def test_init(discipline):
    """Check the instantiation of the discipline."""
    assert discipline.name == "IshigamiDiscipline"
    input_names = ["x1", "x2", "x3"]
    assert list(discipline.input_grammar.names) == input_names
    assert list(discipline.output_grammar.names) == ["y"]
    assert discipline.default_inputs == {name: array([0.0]) for name in input_names}


def test_execute(discipline, input_values):
    """Check the output value of the discipline."""
    assert discipline.execute(input_values[0])["y"][0] == compute_output(
        input_values[1]
    )


def test_gradient(discipline, input_values):
    """Check the gradient of the discipline."""
    gradient = discipline.linearize(input_data=input_values[0], force_all=True)["y"]
    assert_equal(
        array([gradient[name][0, 0] for name in discipline.input_grammar.names]),
        compute_gradient(input_values[1]),
    )
