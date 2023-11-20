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
"""Test for the Splits collection."""

from __future__ import annotations

import pytest
from numpy import array
from numpy.testing import assert_array_equal

from gemseo.mlearning.resampling.split import Split


@pytest.fixture(scope="module")
def split() -> Split:
    """A train-test split."""
    return Split(array([1, 2]), array([3, 4]))


def test_fields(split):
    """Check the fields of a train-test split."""
    assert_array_equal(split.train, array([1, 2]))
    assert_array_equal(split.test, array([3, 4]))


def test_eq(split):
    """Check the method Split.__eq__."""
    assert split == Split(array([1, 2]), array([3, 4]))
    assert split != Split(array([1, 2]), array([3, 5]))
