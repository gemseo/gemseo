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
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Test the tolerance interval for lognormal distributions."""
from __future__ import annotations

from math import exp

import pytest
from gemseo.uncertainty.statistics.tolerance_interval.distribution import (
    ToleranceIntervalSide,
)
from gemseo.uncertainty.statistics.tolerance_interval.lognormal import (
    LogNormalToleranceInterval,
)
from scipy.special import erfinv


def test_lognormal_quantile_both():
    """Check the bounds of two-sided TI for the standard log-normal distribution."""
    tolerance_interval = LogNormalToleranceInterval(
        1000000, mean=0.0, std=1.0, location=0.5
    )
    lower, upper = tolerance_interval.compute(0.95, 0.9)
    assert pytest.approx(lower, 0.01) == exp(2**0.5 * erfinv(2 * 0.025 - 1)) + 0.5
    assert pytest.approx(upper, 0.01) == exp(2**0.5 * erfinv(2 * 0.975 - 1)) + 0.5


def test_lognormal_quantile_lower():
    """Check the bounds of lower-sided TI for the standard log-normal distribution."""
    tolerance_interval = LogNormalToleranceInterval(
        1000000, mean=0.0, std=1.0, location=0.5
    )
    lower, upper = tolerance_interval.compute(
        0.975, 0.9, side=ToleranceIntervalSide.LOWER
    )
    assert pytest.approx(lower, 0.01) == exp(2**0.5 * erfinv(2 * 0.025 - 1)) + 0.5


def test_lognormal_quantile_upper():
    """Check the bounds of upper-sided TI for the standard log-normal distribution."""
    tolerance_interval = LogNormalToleranceInterval(
        1000000, mean=0.0, std=1.0, location=0.5
    )
    lower, upper = tolerance_interval.compute(
        0.975, 0.9, side=ToleranceIntervalSide.UPPER
    )
    assert pytest.approx(upper, 0.01) == exp(2**0.5 * erfinv(2 * 0.975 - 1)) + 0.5
