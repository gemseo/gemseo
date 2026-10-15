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
"""Tests for the assessment of the constraints at a point."""

from __future__ import annotations

import pickle

import pytest
from numpy import inf

from gemseo.optimization.core.constraint_assessment import ConstraintAssessment
from gemseo.optimization.core.constraint_status import ConstraintStatus
from gemseo.optimization.core.constraints_assessment import ConstraintsAssessment


@pytest.mark.parametrize(
    (
        "name_to_assessment",
        "are_satisfied",
        "squared_violation",
        "max_violation",
        "n_violated_components",
        "missing_names",
    ),
    [
        pytest.param({}, True, 0.0, 0.0, 0, (), id="no_constraint"),
        pytest.param(
            {
                "c1": ConstraintAssessment(ConstraintStatus.SATISFIED),
                "c2": ConstraintAssessment(ConstraintStatus.SATISFIED),
            },
            True,
            0.0,
            0.0,
            0,
            (),
            id="all_satisfied",
        ),
        pytest.param(
            {
                "c1": ConstraintAssessment(ConstraintStatus.SATISFIED),
                "c2": ConstraintAssessment(ConstraintStatus.VIOLATED, 4.0, 2.0, 1),
                "c3": ConstraintAssessment(ConstraintStatus.MISSING, inf, inf),
            },
            False,
            inf,
            inf,
            # The missing constraint does not contribute a violated component.
            1,
            ("c3",),
            id="mixed_statuses",
        ),
    ],
)
def test_constraints_assessment(
    name_to_assessment,
    are_satisfied,
    squared_violation,
    max_violation,
    n_violated_components,
    missing_names,
) -> None:
    """Check the properties of ConstraintsAssessment."""
    assessment = ConstraintsAssessment(name_to_assessment)
    assert assessment.are_satisfied is are_satisfied
    assert assessment.squared_violation == squared_violation
    assert assessment.max_violation == max_violation
    assert assessment.n_violated_components == n_violated_components
    assert assessment.missing_names == missing_names


def test_name_to_assessment_is_read_only() -> None:
    """Check that the assessments of the constraints cannot be modified."""
    assessment = ConstraintsAssessment({
        "g": ConstraintAssessment(ConstraintStatus.SATISFIED)
    })
    with pytest.raises(TypeError):
        assessment.name_to_assessment["g"] = ConstraintAssessment(
            ConstraintStatus.VIOLATED, 1.0, 1.0, 1
        )


@pytest.fixture
def pickled_assessment() -> ConstraintsAssessment:
    """A ConstraintsAssessment after a pickle round-trip."""
    return pickle.loads(
        pickle.dumps(
            ConstraintsAssessment({
                "c1": ConstraintAssessment(ConstraintStatus.SATISFIED),
                "c2": ConstraintAssessment(ConstraintStatus.VIOLATED, 4.0, 2.0, 1),
            })
        )
    )


def test_constraints_assessment_pickle_equal(pickled_assessment) -> None:
    """Check that a pickled ConstraintsAssessment is equal to the original."""
    assert pickled_assessment == ConstraintsAssessment({
        "c1": ConstraintAssessment(ConstraintStatus.SATISFIED),
        "c2": ConstraintAssessment(ConstraintStatus.VIOLATED, 4.0, 2.0, 1),
    })


def test_constraints_assessment_pickle_read_only(pickled_assessment) -> None:
    """Check that a pickled ConstraintsAssessment is still read-only."""
    with pytest.raises(TypeError):
        pickled_assessment.name_to_assessment["c3"] = ConstraintAssessment(
            ConstraintStatus.SATISFIED
        )
