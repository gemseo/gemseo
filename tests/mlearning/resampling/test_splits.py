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

from collections.abc import Collection

import pytest
from numpy import array

from gemseo.mlearning.resampling.split import Split
from gemseo.mlearning.resampling.splits import Splits


@pytest.fixture(scope="module")
def first_split() -> Split:
    """The first train-test split."""
    return Split(array([1, 2]), array([3]))


@pytest.fixture(scope="module")
def second_split() -> Split:
    """The second train-test split."""
    return Split(array([2, 3]), array([1]))


@pytest.fixture(scope="module")
def splits(first_split, second_split) -> Splits:
    """A collection of train-test splits."""
    return Splits(first_split, second_split)


def test_collection():
    """Check that Splits is a subclass of collections.abc.Collection."""
    assert issubclass(Splits, Collection)


def test_len(splits):
    """Check the length of a collection of train-test splits."""
    assert len(splits) == 2


def test_iter(splits, first_split, second_split):
    """Check the objects returned when iterating Splits."""
    for split, one_split in zip(splits, (first_split, second_split)):
        assert split == one_split


def test_contains(splits, first_split):
    """Check the method Splits.__contains__."""
    assert first_split in splits
    assert Split(array([12]), array([4])) not in splits
