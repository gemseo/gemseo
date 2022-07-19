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
#
# Contributors:
# - Matthias De Lozzo
# - Jean-Christophe Giret
# - François Gallard
# - Antoine DECHAUME
from gemseo.uncertainty.distributions.factory import DistributionFactory


def test_constructor():
    DistributionFactory()


def test_available():
    factory = DistributionFactory()
    distributions = factory.available_distributions
    assert "OTNormalDistribution" in distributions


def test_is_available():
    factory = DistributionFactory()
    assert factory.is_available("OTNormalDistribution")


def test_create():
    factory = DistributionFactory()
    assert factory.create("OTNormalDistribution", variable="x")
