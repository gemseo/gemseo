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
from numpy import array_equal
from numpy.testing import assert_equal

from gemseo.problems.scalable.parametric.core.disciplines.scalable_discipline import (
    Coefficients,
)
from gemseo.problems.scalable.parametric.core.disciplines.scalable_discipline import (
    ScalableDiscipline as CoreScalableDiscipline,
)
from gemseo.problems.scalable.parametric.disciplines.scalable_discipline import (
    ScalableDiscipline,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


@pytest.fixture(scope="module")
def default_input_values() -> dict[str, NDArray[float]]:
    """The default input values."""
    return {
        "x_0": array([0.0, 0.0]),
        "x_1": array([0.0, 0.0, 0.0]),
        "y_2": array([0.0, 0.0, 0.0]),
    }


@pytest.fixture(scope="module")
def coefficients() -> Coefficients:
    """The coefficients of a scalable discipline."""
    return Coefficients(
        array([1.0, 2.0]),
        array([[-1.0, 0.0], [0.0, -1.0]]),
        array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
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


@pytest.fixture(scope="module")
def core_scalable_discipline(
    coefficients, default_input_values
) -> CoreScalableDiscipline:
    """The discipline wrapped by the scalable discipline."""
    return CoreScalableDiscipline(
        1,
        coefficients.a_i,
        coefficients.D_i0,
        coefficients.D_ii,
        coefficients.C_ij,
        **default_input_values,
    )


def test_wrapped_discipline(scalable_discipline):
    """Check that ScalableDiscipline is composed of a CoreScalableDiscipline."""
    assert CoreScalableDiscipline == scalable_discipline._CORE_DISCIPLINE_CLASS


def test_execution(scalable_discipline, core_scalable_discipline):
    """Check the execution of ScalableDiscipline."""
    core_output_data = core_scalable_discipline(
        x_0=array([1.0, 2.0]),
        x_i=array([1.0, 2.0, 1.0]),
        y_2=array([1.0, 2.0, 3.0]),
    )
    scalable_discipline.execute({
        "x_0": array([1.0, 2.0]),
        "x_1": array([1.0, 2.0, 1.0]),
        "y_2": array([1.0, 2.0, 3.0]),
    })
    assert_equal(dict(scalable_discipline.get_output_data()), core_output_data)


def test_differentiation(scalable_discipline, core_scalable_discipline):
    """Check the differentiation of ScalableDiscipline."""
    input_data = {
        "x_0": array([1.0, 2.0]),
        "x_1": array([1.0, 2.0, 1.0]),
        "y_2": array([1.0, 2.0, 3.0]),
    }
    assert_equal(
        scalable_discipline.linearize(
            input_data,
            compute_all_jacobians=True,
        ),
        core_scalable_discipline(
            x_0=input_data["x_0"],
            x_i=input_data["x_1"],
            y_2=input_data["y_2"],
            compute_jacobian=True,
        ),
    )
    assert scalable_discipline.check_jacobian(
        input_data, derr_approx=scalable_discipline.LinearizationMode.COMPLEX_STEP
    )


def test_random_variables(default_input_values, coefficients):
    """Check the use of random variables."""
    scalable_discipline = ScalableDiscipline(
        1,
        coefficients.a_i,
        coefficients.D_i0,
        coefficients.D_ii,
        coefficients.C_ij,
        u_1=array([0.0]),
        **default_input_values,
    )
    assert scalable_discipline._discipline.input_names == ["u_1", "x_0", "x_1", "y_2"]
    assert_equal(
        scalable_discipline.execute()["y_1"],
        scalable_discipline.execute({"u_1": array([0.0])})["y_1"],
    )
    assert not array_equal(
        scalable_discipline.execute()["y_1"],
        scalable_discipline.execute({"u_1": array([1.0])})["y_1"],
    )
