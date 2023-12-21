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

from typing import TYPE_CHECKING

import pytest
from numpy import array
from numpy.testing import assert_equal

from gemseo.problems.scalable.parametric.core.disciplines.main_discipline import (
    MainDiscipline as CoreMainDiscipline,
)
from gemseo.problems.scalable.parametric.disciplines.main_discipline import (
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


@pytest.fixture(scope="module")
def core_main_discipline(default_input_values) -> CoreMainDiscipline:
    """The discipline wrapped by the main discipline."""
    return CoreMainDiscipline(
        array([1.0, 2.0, 4.0]), array([-1.0, -2.0]), **default_input_values
    )


def test_wrapped_discipline(main_discipline):
    """Check that MainDiscipline is composed of a CoreMainDiscipline."""
    assert CoreMainDiscipline == main_discipline._CORE_DISCIPLINE_CLASS


def test_execution(main_discipline, core_main_discipline):
    """Check the execution of MainDiscipline."""
    main_discipline.execute({
        "x_0": array([1.0, 2.0, 1.0, 2.0]),
        "y_2": array([1.0, 2.0]),
    })
    assert_equal(
        dict(main_discipline.get_output_data()),
        core_main_discipline(x_0=array([1.0, 2.0, 1.0, 2.0]), y_2=array([1.0, 2.0])),
    )


def test_differentiation(main_discipline, core_main_discipline):
    """Check the differentiation of MainDiscipline."""
    input_data = {"x_0": array([1.0, 2.0, 1.0, 2.0]), "y_2": array([1.0, 2.0])}
    assert_equal(
        main_discipline.linearize(
            input_data,
            compute_all_jacobians=True,
        ),
        core_main_discipline(
            x_0=input_data["x_0"],
            y_2=input_data["y_2"],
            compute_jacobian=True,
        ),
    )
    assert main_discipline.check_jacobian(
        input_data, derr_approx=main_discipline.LinearizationMode.COMPLEX_STEP
    )
