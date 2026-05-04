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
"""Test the factory used to create instances of BaseToleranceInterval."""

from __future__ import annotations

import re

from gemseo.uncertainty.statistics.tolerance_interval.factory import (
    TOLERANCE_INTERVAL_FACTORY,
)
from gemseo.uncertainty.statistics.tolerance_interval.normal import (
    NormalToleranceInterval,
)
from gemseo.utils.testing.helpers import assert_exception


def test_create() -> None:
    """Check the creation of a BaseToleranceInterval from the
    ToleranceIntervalFactory."""
    tolerance_interval = TOLERANCE_INTERVAL_FACTORY.create(
        "NormalToleranceInterval", 100000, 0, 1
    )
    assert isinstance(tolerance_interval, NormalToleranceInterval)
    assert tolerance_interval._NormalToleranceInterval__mean == 0.0
    assert tolerance_interval._NormalToleranceInterval__std == 1.0


def test_create_fail(snapshot) -> None:
    """Check the creation of a BaseToleranceInterval from the
    ToleranceIntervalFactory."""
    re.escape(
        "The class WrongName is not available; "
        "the available ones are: ExponentialToleranceInterval, "
        "LogNormalToleranceInterval, NormalToleranceInterval, "
        "UniformToleranceInterval, WeibullMinToleranceInterval, "
        "WeibullToleranceInterval."
    )

    with assert_exception(ImportError, snapshot):
        TOLERANCE_INTERVAL_FACTORY.create("WrongName", 100000, 0, 1)

    with assert_exception(TypeError, snapshot):
        TOLERANCE_INTERVAL_FACTORY.create("NormalToleranceInterval", 100000)
