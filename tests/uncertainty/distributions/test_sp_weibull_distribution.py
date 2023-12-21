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
"""Tests for the class SPWeibullDistribution."""

from __future__ import annotations

from scipy.stats import weibull_max
from scipy.stats import weibull_min

from gemseo.uncertainty.distributions.scipy.weibull import SPWeibullDistribution


def test_default_distribution():
    """Check the default Weibull distribution."""
    distribution = SPWeibullDistribution()
    assert distribution.variable_name == distribution.DEFAULT_VARIABLE_NAME
    marginal = distribution.marginals[0]
    assert marginal.mean() == weibull_min.mean(1, loc=0, scale=1)
    assert marginal.var() == weibull_min.var(1, loc=0, scale=1)


def test_custom():
    """Check a custom Weibull distribution."""
    distribution = SPWeibullDistribution(
        "u", location=2.0, scale=3.0, shape=4.0, use_weibull_min=False
    )
    assert distribution.variable_name == "u"
    marginal = distribution.marginals[0]
    assert marginal.mean() == weibull_max.mean(4, loc=2, scale=3)
    assert marginal.var() == weibull_max.var(4, loc=2, scale=3)
