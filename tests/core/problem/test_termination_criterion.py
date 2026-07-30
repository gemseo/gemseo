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
"""Tests for the generic termination criteria."""

from __future__ import annotations

import pytest

from gemseo.core.problem.termination_criterion import DesvarIsNan
from gemseo.core.problem.termination_criterion import FtolReached
from gemseo.core.problem.termination_criterion import FunctionIsNan
from gemseo.core.problem.termination_criterion import MaxIterReachedException
from gemseo.core.problem.termination_criterion import MaxTimeReached
from gemseo.core.problem.termination_criterion import TerminationCriterion
from gemseo.core.problem.termination_criterion import XtolReached


@pytest.mark.parametrize(
    ("criterion_class", "expected_message"),
    [
        (TerminationCriterion, ""),
        (MaxIterReachedException, "Maximum number of iterations reached. "),
        (
            FunctionIsNan,
            (
                "Function value or gradient or constraint is NaN, "
                "and problem.stop_if_nan is set to True. "
            ),
        ),
        (DesvarIsNan, "Design variables are NaN. "),
        (
            XtolReached,
            (
                "Successive iterates of the design variables "
                "are closer than xtol_rel or xtol_abs. "
            ),
        ),
        (
            FtolReached,
            (
                "Successive iterates of the objective function "
                "are closer than ftol_rel or ftol_abs. "
            ),
        ),
        (MaxTimeReached, "Maximum time reached. "),
    ],
)
def test_default_message(
    criterion_class: type[TerminationCriterion], expected_message: str
) -> None:
    """Check the default message of the termination criteria."""
    assert criterion_class().message == expected_message


def test_message_from_argument() -> None:
    """Check that the first exception argument overrides the default message."""
    assert MaxTimeReached("Maximum time reached: 10 seconds. ").message == (
        "Maximum time reached: 10 seconds. "
    )
