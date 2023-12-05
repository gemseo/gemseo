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
"""Tests for SPDistributionFactory."""

from __future__ import annotations

import pytest

from gemseo.uncertainty.distributions.factory import DistributionFactory
from gemseo.uncertainty.distributions.scipy.distribution import SPDistribution
from gemseo.uncertainty.distributions.scipy.factory import SPDistributionFactory


@pytest.fixture(scope="module")
def factory() -> SPDistributionFactory:
    """A factory of SPDistribution objects."""
    return SPDistributionFactory()


def test_inheritance(factory):
    """Check that a SPDistributionFactory is a DistributionFactory."""
    assert isinstance(factory, DistributionFactory)


def test_class_names(factory):
    """Check that the SPDistributionFactory can only create SPDistribution objects."""
    class_names = factory.class_names
    assert {"SPDistribution", "SPNormalDistribution"}.issubset(class_names)
    for class_name in class_names:
        assert issubclass(factory.get_class(class_name), SPDistribution)
