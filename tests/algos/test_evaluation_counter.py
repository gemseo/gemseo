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

from gemseo.algos.evaluation_counter import EvaluationCounter
from gemseo.utils.testing.helpers import assert_exception


def test_default():
    """Check the values of the EvaluationCounter with default settings."""
    counter = EvaluationCounter()
    assert counter.current == 0
    assert counter.maximum == 0
    assert not counter.maximum_is_reached


def test_custom():
    """Check the values of the EvaluationCounter with custom settings."""
    counter = EvaluationCounter(current=1, maximum=3)
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
    assert EvaluationCounter(current=3, maximum=3).maximum_is_reached


def test_error(snapshot):
    """Check that the current value cannot be greater than the maximum."""
    with assert_exception(ValueError, snapshot):
        EvaluationCounter(current=3, maximum=2)
