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
"""Check unsuitability_reason."""
from __future__ import annotations

import pytest
from gemseo.algos._unsuitability_reason import _UnsuitabilityReason


@pytest.mark.parametrize(
    "name,value",
    [
        ("NO_REASON", ""),
        ("EMPTY_DESIGN_SPACE", "the design space is empty"),
        ("NOT_SYMMETRIC", "the left-hand side of the problem is not symmetric"),
        (
            "NOT_POSITIVE_DEFINITE",
            "the left-hand side of the problem is not positive definite",
        ),
        (
            "NOT_LINEAR_OPERATOR",
            "the left-hand side of the problem is not a linear operator",
        ),
        ("NON_LINEAR_PROBLEM", "it does not handle non-linear problems"),
        ("INEQUALITY_CONSTRAINTS", "it does not handle inequality constraints"),
        ("EQUALITY_CONSTRAINTS", "it does not handle equality constraints"),
        (
            "SMALL_DIMENSION",
            (
                "the dimension of the problem is lower "
                "than the minimum dimension it can handle"
            ),
        ),
    ],
)
def test_reason_values(name, value):
    """Check the value of a _UnsuitabilityReason."""
    assert _UnsuitabilityReason[name].value == value


def test_str():
    """Check that the string representation of an _UnsuitabilityReason is its value."""
    reason = _UnsuitabilityReason.EMPTY_DESIGN_SPACE
    assert str(reason) == reason.value


def test_bool():
    """Check that a _UnsuitabilityReason is False only if equal to NO_REASON."""
    for reason in _UnsuitabilityReason:
        if reason == _UnsuitabilityReason.NO_REASON:
            assert not reason
        else:
            assert reason
