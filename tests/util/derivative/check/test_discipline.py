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
"""Tests for DisciplineJacobianChecker."""

from __future__ import annotations

import pytest
from numpy.testing import assert_allclose

from gemseo.discipline.analytic import AnalyticDiscipline
from gemseo.util.derivative.check.discipline import DisciplineJacobianChecker


@pytest.fixture
def discipline() -> AnalyticDiscipline:
    """A linear analytic discipline with an exact Jacobian."""
    return AnalyticDiscipline({"y": "2*x"}, name="d")


def test_check_passes(discipline) -> None:
    """Check that a correct analytical Jacobian is validated."""
    assert DisciplineJacobianChecker(discipline).check()


def test_discipline_linearized_after_check(discipline) -> None:
    """check() leaves the discipline linearized at the check point.

    Only the perturbations of the numerical approximation are reverted;
    the analytic Jacobian and the data of the linearization are kept.
    """
    DisciplineJacobianChecker(discipline).check()

    # The analytic Jacobian is available.
    assert_allclose(discipline.jac["y"]["x"], [[2.0]])
    # The input data is the check point, not a perturbed point.
    assert_allclose(
        discipline.io.input_data["x"], discipline.io.input_grammar.defaults["x"]
    )


def test_approximation_perturbations_reverted(discipline) -> None:
    """The numerical approximation does not corrupt the linearized state.

    Linearizing then checking leaves the analytic Jacobian and the data
    untouched by the perturbed executions of the approximation.
    """
    discipline.add_differentiated_inputs(["x"])
    discipline.add_differentiated_outputs(["y"])
    discipline.linearize()
    jac = dict(discipline.jac)
    input_data = dict(discipline.io.input_data)
    output_data = dict(discipline.io.output_data)

    DisciplineJacobianChecker(discipline).check()

    assert_allclose(discipline.jac["y"]["x"], jac["y"]["x"])
    assert dict(discipline.io.input_data) == input_data
    assert dict(discipline.io.output_data) == output_data
