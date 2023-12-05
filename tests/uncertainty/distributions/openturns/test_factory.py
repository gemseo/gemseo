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
"""Tests for OTDistributionFactory."""

from __future__ import annotations

import pytest

from gemseo.uncertainty.distributions.factory import DistributionFactory
from gemseo.uncertainty.distributions.openturns.distribution import OTDistribution
from gemseo.uncertainty.distributions.openturns.factory import OTDistributionFactory


@pytest.fixture(scope="module")
def factory() -> OTDistributionFactory:
    """A factory of OTDistribution objects."""
    return OTDistributionFactory()


def test_inheritance(factory):
    """Check that a OTDistributionFactory is a DistributionFactory."""
    assert isinstance(factory, DistributionFactory)


def test_class_names(factory):
    """Check that the OTFactoryFactory can only create OTDistribution objects."""
    class_names = factory.class_names
    assert {"OTDistribution", "OTNormalDistribution"}.issubset(class_names)
    for class_name in class_names:
        assert issubclass(factory.get_class(class_name), OTDistribution)
