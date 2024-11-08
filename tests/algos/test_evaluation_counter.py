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

import re

import pytest

from gemseo.algos.evaluation_counter import EvaluationCounter


def test_default():
    """Check the values of the EvaluationCounter with default settings."""
    counter = EvaluationCounter()
    assert counter.current == 0
    assert counter.maximum == 0
    assert not counter.maximum_is_reached


def test_custom():
    """Check the values of the EvaluationCounter with custom settings."""
    counter = EvaluationCounter(1, 3)
    assert counter.current == 1
    assert counter.maximum == 3
    assert not counter.maximum_is_reached


def test_setters():
    """Check that the current and maximum values can be set after instantiation."""
    counter = EvaluationCounter()
    counter.current = 1
    counter.maximum = 3
    assert counter.current == 1
    assert counter.maximum == 3
    assert not counter.maximum_is_reached


def test_maximum_is_reached():
    """Check that the property maximum_is_reached."""
    assert EvaluationCounter(3, 3).maximum_is_reached


def test_error():
    """Check that the current value cannot be greater than the maximum."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The current value (3) of the evaluation counter must be "
            "less than or equal to the maximum number of evaluations (2)."
        ),
    ):
        EvaluationCounter(3, 2)
