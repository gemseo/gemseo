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


def test_default_distribution():
    """Check the default Weibull distribution."""
    distribution = OTWeibullDistribution()
    assert distribution.variable_name == distribution.DEFAULT_VARIABLE_NAME
    marginal = distribution.marginals[0]
    assert isinstance(marginal, WeibullMin)
    assert marginal.getParameter() == [1, 1, 0]


def test_custom():
    """Check a custom Weibull distribution."""
    distribution = OTWeibullDistribution(
        "u",
        location=2.0,
        scale=3.0,
        shape=4.0,
        use_weibull_min=False,
    )
    assert distribution.variable_name == "u"
    marginal = distribution.marginals[0]
    assert isinstance(marginal, WeibullMax)
    assert marginal.getParameter() == [3.0, 4.0, 2.0]
