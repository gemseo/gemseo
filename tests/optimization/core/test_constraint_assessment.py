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
"""Tests for the assessment of a constraint at a value."""

from __future__ import annotations

import pytest
from numpy import array
from numpy import inf
from numpy import nan

from gemseo.core.function.array_function import ArrayFunction
from gemseo.optimization.constraint_tolerances import ConstraintTolerances
from gemseo.optimization.core.constraint_assessment import ConstraintAssessment
from gemseo.optimization.core.constraint_status import ConstraintStatus
from gemseo.optimization.core.constraints import Constraints
from gemseo.space.design import DesignSpace

_EQ = ArrayFunction.ConstraintType.EQ
_INEQ = ArrayFunction.ConstraintType.INEQ
_TOLERANCE = 1.0


@pytest.mark.parametrize(
    ("value", "constraint_type", "expected"),
    [
        # Satisfied, strictly and exactly at the tolerance.
        (
            array([-2.0, 1.0]),
            _INEQ,
            ConstraintAssessment(ConstraintStatus.SATISFIED),
        ),
        (
            array([1.0]),
            _EQ,
            ConstraintAssessment(ConstraintStatus.SATISFIED),
        ),
        # A scalar, not an array.
        (
            0.5,
            _INEQ,
            ConstraintAssessment(ConstraintStatus.SATISFIED),
        ),
        # A complex value, e.g. from the complex-step method,
        # whose real part is at the tolerance.
        (
            array([1.0 + 1.0j]),
            _INEQ,
            ConstraintAssessment(ConstraintStatus.SATISFIED),
        ),
        (
            array([1.0 + 1.0j]),
            _EQ,
            ConstraintAssessment(ConstraintStatus.SATISFIED),
        ),
        # Violated: one component out of two.
        (
            array([-2.0, 3.0]),
            _INEQ,
            ConstraintAssessment(ConstraintStatus.VIOLATED, 4.0, 2.0, 1),
        ),
        # Violated: both components.
        (
            array([3.0, 4.0]),
            _INEQ,
            ConstraintAssessment(ConstraintStatus.VIOLATED, 13.0, 3.0, 2),
        ),
        # Violated equality constraint.
        (
            array([-2.5]),
            _EQ,
            ConstraintAssessment(ConstraintStatus.VIOLATED, 2.25, 1.5, 1),
        ),
        # Partial NaN: only the NaN component is violated.
        (
            array([nan, -1.0]),
            _INEQ,
            ConstraintAssessment(ConstraintStatus.VIOLATED, inf, inf, 1),
        ),
        # Full NaN.
        (
            array([nan, nan]),
            _INEQ,
            ConstraintAssessment(ConstraintStatus.VIOLATED, inf, inf, 2),
        ),
    ],
)
def test_from_value(value, constraint_type, expected) -> None:
    """Check ConstraintAssessment.from_value on satisfied, violated and NaN values."""
    assessment = ConstraintAssessment.from_value(value, constraint_type, _TOLERANCE)
    assert assessment.status is expected.status
    assert assessment.squared_violation == pytest.approx(expected.squared_violation)
    assert assessment.max_violation == pytest.approx(expected.max_violation)
    assert assessment.n_violated_components == expected.n_violated_components


@pytest.mark.parametrize("constraint_type", [_EQ, _INEQ])
def test_from_value_with_infinite_tolerance(constraint_type) -> None:
    """Check that an infinite value satisfies an infinite tolerance.

    This is consistent with `Constraints.is_constraint_satisfied`.
    """
    value = array([inf])
    constraints = Constraints(
        DesignSpace(), ConstraintTolerances(inequality=inf, equality=inf)
    )

    assessment = ConstraintAssessment.from_value(value, constraint_type, inf)
    assert assessment == ConstraintAssessment(ConstraintStatus.SATISFIED)

    is_satisfied = constraints.is_constraint_satisfied(constraint_type, value)
    assert is_satisfied


def test_repr() -> None:
    """Check the representation of a ConstraintAssessment."""
    assessment = ConstraintAssessment(ConstraintStatus.VIOLATED, 0.35988, 0.5999, 1)
    expected = (
        "ConstraintAssessment(violated, squared_violation=0.35988, "
        "max_violation=0.5999, n_violated_components=1)"
    )
    assert repr(assessment) == expected
