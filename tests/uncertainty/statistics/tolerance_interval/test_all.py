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
"""Test the package implementing tolerance intervals."""
from __future__ import annotations

import pytest
from gemseo.uncertainty.statistics.tolerance_interval.distribution import (
    ToleranceIntervalSide,
)
from gemseo.uncertainty.statistics.tolerance_interval.exponential import (
    ExponentialToleranceInterval,
)
from gemseo.uncertainty.statistics.tolerance_interval.lognormal import (
    LogNormalToleranceInterval,
)
from gemseo.uncertainty.statistics.tolerance_interval.normal import (
    NormalToleranceInterval,
)
from gemseo.uncertainty.statistics.tolerance_interval.uniform import (
    UniformToleranceInterval,
)
from gemseo.uncertainty.statistics.tolerance_interval.weibull import (
    WeibullToleranceInterval,
)

DISTRIBUTIONS = {
    "Weibull": WeibullToleranceInterval(10, scale=1.0, shape=2.0, location=0.0),
    "Normal": NormalToleranceInterval(10, mean=0.0, std=1),
    "LogNormal": LogNormalToleranceInterval(10, mean=0.0, std=1, location=0.0),
    "Exponential": ExponentialToleranceInterval(10, rate=1.0, location=0.0),
    "Uniform": UniformToleranceInterval(10, minimum=0.0, maximum=1.0),
}


@pytest.mark.parametrize("side", ToleranceIntervalSide)
@pytest.mark.parametrize("distribution", DISTRIBUTIONS.keys())
def test_tolerance_interval(side, distribution):
    """Check that the lower bound is lower than the upper one."""
    tolerance_interval = DISTRIBUTIONS[distribution]
    lower, upper = tolerance_interval.compute(0.90, side=side)
    assert lower < upper


@pytest.mark.parametrize("side", ToleranceIntervalSide)
@pytest.mark.parametrize("distribution", DISTRIBUTIONS.keys())
def test_tolerance_interval_coverage_lower(side, distribution):
    """Check that the lower bound is lower when the coverage is higher."""
    tolerance_interval = DISTRIBUTIONS[distribution]
    lower_with_95_coverage, _ = tolerance_interval.compute(coverage=0.95, side=side)
    lower_with_90_coverage, _ = tolerance_interval.compute(coverage=0.90, side=side)
    assert lower_with_95_coverage <= lower_with_90_coverage


@pytest.mark.parametrize("side", ToleranceIntervalSide)
@pytest.mark.parametrize("distribution", DISTRIBUTIONS.keys())
def test_tolerance_interval_coverage_upper(side, distribution):
    """Check that the upper bound is higher when the coverage is higher."""
    tolerance_interval = DISTRIBUTIONS[distribution]
    _, upper_with_95_coverage = tolerance_interval.compute(coverage=0.95, side=side)
    _, upper_with_90_coverage = tolerance_interval.compute(coverage=0.90, side=side)
    assert upper_with_90_coverage <= upper_with_95_coverage


@pytest.mark.parametrize("side", ToleranceIntervalSide)
@pytest.mark.parametrize("distribution", DISTRIBUTIONS.keys())
def test_tolerance_interval_confidence_lower(side, distribution):
    """Check that the lower bound is lower when the confidence is higher."""
    tolerance_interval = DISTRIBUTIONS[distribution]
    lower_with_95_confidence, _ = tolerance_interval.compute(
        coverage=0.90, confidence=0.95, side=side
    )
    lower_with_90_confidence, _ = tolerance_interval.compute(
        coverage=0.90, confidence=0.90, side=ToleranceIntervalSide.LOWER
    )
    assert lower_with_95_confidence <= lower_with_90_confidence


@pytest.mark.parametrize("side", ToleranceIntervalSide)
@pytest.mark.parametrize("distribution", DISTRIBUTIONS.keys())
def test_tolerance_interval_confidence_upper(side, distribution):
    """Check that the upper bound is higher when the confidence is higher."""
    tolerance_interval = DISTRIBUTIONS[distribution]
    _, upper_with_95_confidence = tolerance_interval.compute(
        coverage=0.90, confidence=0.95, side=side
    )
    _, upper_with_90_confidence = tolerance_interval.compute(
        coverage=0.90, confidence=0.90, side=side
    )
    assert upper_with_90_confidence <= upper_with_95_confidence


@pytest.mark.parametrize("distribution_name", ["Uniform", "Normal"])
def test_tolerance_interval_incorrect_side(distribution_name):
    """Check that an error is raised if the type of tolerance interval is incorrect."""
    with pytest.raises(
        ValueError, match="The type of tolerance interval is incorrect."
    ):
        DISTRIBUTIONS[distribution_name].compute(0.9, side="incorrect_side")
