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
"""Module containing a factory to create an instance of :class:`.OTDistribution`."""

from __future__ import annotations

from gemseo.uncertainty.distributions.factory import DistributionFactory
from gemseo.uncertainty.distributions.openturns.distribution import OTDistribution


class OTDistributionFactory(DistributionFactory):
    """Factory to create a :class:`.OTDistribution` from its class name."""

    _CLASS = OTDistribution
    _MODULE_NAMES = ("gemseo.uncertainty.distributions.openturns",)
