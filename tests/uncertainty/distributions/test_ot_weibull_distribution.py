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
"""Tests for the class OTWeibullDistribution."""

from __future__ import annotations

from openturns import WeibullMax
from openturns import WeibullMin

from gemseo.uncertainty.distributions.openturns.weibull import OTWeibullDistribution
from gemseo.uncertainty.distributions.openturns.weibull_settings import (
    OTWeibullDistribution_Settings,
)


def test_default_distribution() -> None:
    """Check the default Weibull distribution."""
    distribution = OTWeibullDistribution()
    distribution = distribution.distribution
    assert isinstance(distribution, WeibullMin)
    assert distribution.getParameter() == [1, 1, 0]


def test_custom() -> None:
    """Check a custom Weibull distribution."""
    distribution = OTWeibullDistribution(
        OTWeibullDistribution_Settings(
            location=2.0, scale=3.0, shape=4.0, use_weibull_min=False
        )
    )
    distribution = distribution.distribution
    assert isinstance(distribution, WeibullMax)
    assert distribution.getParameter() == [3.0, 4.0, 2.0]
