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
"""Test for the class DistributionFactory."""

from __future__ import annotations

import re

import pytest

from gemseo.uncertainty.distributions.factory import DistributionFactory
from gemseo.uncertainty.distributions.openturns.joint import OTJointDistribution
from gemseo.uncertainty.distributions.openturns.normal import OTNormalDistribution


@pytest.fixture
def distribution_factory() -> DistributionFactory:
    """A distribution factory."""
    return DistributionFactory()


def test_is_available(distribution_factory) -> None:
    """Check is_available()."""
    assert distribution_factory.is_available("OTNormalDistribution")
    assert not distribution_factory.is_available("foo")


def test_create_marginal_distribution(distribution_factory) -> None:
    """Check create_marginal_distribution() to instantiate a marginal distribution."""
    distribution = distribution_factory.create_marginal_distribution(
        "OTNormalDistribution"
    )
    assert isinstance(distribution, OTNormalDistribution)
    assert (
        distribution_factory.create_marginal_distribution == distribution_factory.create
    )


def test_create_joint_distribution(distribution_factory) -> None:
    """Check create_joint_distribution() to instantiate a joint probability
    distribution."""
    normal = distribution_factory.create("OTNormalDistribution")
    uniform = distribution_factory.create("OTUniformDistribution")
    joint_distribution = distribution_factory.create_joint_distribution(
        distributions=[normal, uniform]
    )
    assert isinstance(joint_distribution, OTJointDistribution)
    assert joint_distribution.marginals[0] == normal
    assert joint_distribution.marginals[1] == uniform


def test_create_joint_distribution_with_different_identifiers(
    distribution_factory,
) -> None:
    """Check create_joint_distribution() with different distribution identifiers."""
    normal = distribution_factory.create("OTNormalDistribution")
    uniform = distribution_factory.create("SPUniformDistribution")
    with pytest.raises(
        ValueError,
        match=re.escape(
            "A joint probability distribution cannot mix distributions "
            "with different identifiers; got OT, SP."
        ),
    ):
        distribution_factory.create_joint_distribution(distributions=[normal, uniform])
