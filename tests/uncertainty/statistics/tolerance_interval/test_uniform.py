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
"""Test the tolerance interval for uniform distributions."""
from __future__ import annotations

import pytest
from gemseo.uncertainty.statistics.tolerance_interval.distribution import (
    ToleranceIntervalSide,
)
from gemseo.uncertainty.statistics.tolerance_interval.uniform import (
    UniformToleranceInterval,
)


def test__quantile_both():
    """Check the bounds of two-sided TI for the standard uniform distribution."""
    tolerance_interval = UniformToleranceInterval(1000000, minimum=0.0, maximum=1.0)
    lower, upper = tolerance_interval.compute(0.95, 0.9)
    assert pytest.approx(lower, 0.001) == 0.025
    assert pytest.approx(upper, 0.001) == 0.975


def test_uniform_quantile_lower():
    """Check the bounds of lower-sided TI for the standard uniform distribution."""
    tolerance_interval = UniformToleranceInterval(1000000, minimum=0.0, maximum=1.0)
    lower, upper = tolerance_interval.compute(
        0.975, 0.9, side=ToleranceIntervalSide.LOWER
    )
    assert pytest.approx(lower, 0.001) == 0.025


def test_uniform_quantile_upper():
    """Check the bounds of upper-sided TI for the standard uniform distribution."""
    tolerance_interval = UniformToleranceInterval(1000000, minimum=0.0, maximum=1.0)
    lower, upper = tolerance_interval.compute(
        0.975, 0.9, side=ToleranceIntervalSide.UPPER
    )
    assert pytest.approx(upper, 0.001) == 0.975
