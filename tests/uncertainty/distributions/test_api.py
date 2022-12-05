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
#    INITIAL AUTHORS - initial API and implementation and/or
#                      initial documentation
#        :author:  Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import pytest
from gemseo.uncertainty.distributions.factory import DistributionFactory
from gemseo.uncertainty.distributions.openturns.composed import OTComposedDistribution
from gemseo.uncertainty.distributions.openturns.normal import OTNormalDistribution


@pytest.fixture
def distribution_factory() -> DistributionFactory:
    """A distribution factory."""
    return DistributionFactory()


def test_available_distributions(distribution_factory):
    """Check the property available_distributions."""
    distributions = distribution_factory.available_distributions
    assert distributions == distribution_factory.factory.classes
    assert "OTNormalDistribution" in distributions


def test_is_available(distribution_factory):
    """Check is_available()."""
    assert distribution_factory.is_available("OTNormalDistribution")
    assert not distribution_factory.is_available("foo")


def test_create_marginal_distribution(distribution_factory):
    """Check create_marginal_distribution() to instantiate a marginal distribution."""
    distribution = distribution_factory.create_marginal_distribution(
        "OTNormalDistribution", variable="x"
    )
    assert isinstance(distribution, OTNormalDistribution)
    assert distribution.variable_name == "x"
    assert (
        distribution_factory.create_marginal_distribution == distribution_factory.create
    )


def test_create_composed_distribution(distribution_factory):
    """Check create_composed_distribution() to instantiate a composed distribution."""
    normal = distribution_factory.create("OTNormalDistribution", variable="x")
    uniform = distribution_factory.create("OTUniformDistribution", variable="y")
    composed = distribution_factory.create_composed_distribution(
        distributions=[normal, uniform], variable="foo"
    )
    assert isinstance(composed, OTComposedDistribution)
    assert composed.marginals[0] == normal
    assert composed.marginals[1] == uniform
    assert composed.variable_name == "foo"


def test_create_composed_distribution_with_different_identifiers(distribution_factory):
    """Check create_composed_distribution() with different distribution identifiers."""
    normal = distribution_factory.create("OTNormalDistribution", variable="x")
    uniform = distribution_factory.create("SPUniformDistribution", variable="y")
    with pytest.raises(
        ValueError,
        match=(
            "A composed probability distribution cannot mix distributions "
            "with different identifiers; got OT, SP."
        ),
    ):
        distribution_factory.create_composed_distribution(
            distributions=[normal, uniform]
        )
