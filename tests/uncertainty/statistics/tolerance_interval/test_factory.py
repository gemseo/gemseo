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
"""Test the factory used to create instances of :class:`.ToleranceInterval`."""
from __future__ import annotations

import re

import pytest
from gemseo.uncertainty.statistics.tolerance_interval.distribution import (
    ToleranceIntervalFactory,
)
from gemseo.uncertainty.statistics.tolerance_interval.normal import (
    NormalToleranceInterval,
)


def test_create():
    """Check the creation of a ToleranceInterval from the ToleranceIntervalFactory."""
    factory = ToleranceIntervalFactory()
    tolerance_interval = factory.create("Normal", 100000, 0, 1)
    assert isinstance(tolerance_interval, NormalToleranceInterval)
    assert tolerance_interval._NormalToleranceInterval__mean == 0.0
    assert tolerance_interval._NormalToleranceInterval__std == 1.0


def test_create_fail():
    """Check the creation of a ToleranceInterval from the ToleranceIntervalFactory."""
    factory = ToleranceIntervalFactory()

    expected = re.escape(
        "Class WrongNameToleranceInterval is not available; \n"
        "available ones are: ExponentialToleranceInterval, LogNormalToleranceInterval, "
        "NormalToleranceInterval, UniformToleranceInterval, "
        "WeibullMinToleranceInterval, WeibullToleranceInterval."
    )

    with pytest.raises(ImportError, match=expected):
        factory.create("WrongName", 100000, 0, 1)

    with pytest.raises(
        TypeError, match="Cannot create NormalToleranceInterval with arguments ()"
    ):
        factory.create("Normal", 100000)
