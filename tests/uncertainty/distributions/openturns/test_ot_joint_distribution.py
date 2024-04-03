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

from typing import TYPE_CHECKING

import pytest
from numpy import allclose
from numpy import array
from numpy import inf
from openturns import NormalCopula

from gemseo.uncertainty.distributions.openturns.joint import OTJointDistribution
from gemseo.uncertainty.distributions.openturns.normal import OTNormalDistribution

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.uncertainty.distributions.openturns.distribution import OTDistribution


@pytest.fixture(scope="module")
def distributions() -> list[OTNormalDistribution]:
    """Two normal distributions."""
    return [OTNormalDistribution(name, dimension=2) for name in ["x1", "x2"]]


@pytest.fixture(scope="module")
def joint_distribution(
    distributions: Sequence[OTDistribution],
) -> OTJointDistribution:
    """A joint probability distribution."""
    return OTJointDistribution(distributions)


@pytest.mark.parametrize("dimensions", [(1,), (1, 1), (2,)])
def test_repr(joint_distribution, dimensions) -> None:
    """Check the string representation of a joint probability distribution."""
    normal = "Normal(mu=0.0, sigma=1.0)"
    normal2 = "Normal[2](mu=0.0, sigma=1.0)"
    distributions = [OTNormalDistribution(dimension=dimensions[0])]
    if sum(dimensions) == 1:
        expected = normal
    elif len(dimensions) == 1:
        expected = f"OTJointDistribution({normal2}; IndependentCopula)"
    else:
        expected = f"OTJointDistribution({normal}, {normal}; IndependentCopula)"
        distributions.append(OTNormalDistribution(dimension=dimensions[1]))

    assert repr(OTJointDistribution(distributions)) == expected


def test_constructor(joint_distribution) -> None:
    assert joint_distribution.dimension == 4
    assert joint_distribution.variable_name == "x1_x2"
    assert joint_distribution.distribution_name == "Composed"
    assert joint_distribution.transformation == "x1_x2"
    assert len(joint_distribution.parameters) == 1
    assert joint_distribution.parameters[0] is None


def test_copula(distributions) -> None:
    """Check the use of an OpenTURNS copula."""
    distribution = OTJointDistribution(distributions, copula=NormalCopula(4))
    assert repr(distribution) == (
        "OTJointDistribution("
        "Normal[2](mu=0.0, sigma=1.0), Normal[2](mu=0.0, sigma=1.0); NormalCopula)"
    )
    assert distribution.distribution.getCopula().getName() == "NormalCopula"


def test_variable_name(distributions) -> None:
    """Check the use of a custom variable name."""
    assert OTJointDistribution(distributions, variable="foo").variable_name == "foo"


def test_str(joint_distribution) -> None:
    """Check the string representation of the joint probability distribution."""
    assert (
        repr(joint_distribution)
        == str(joint_distribution)
        == (
            "OTJointDistribution("
            "Normal[2](mu=0.0, sigma=1.0), "
            "Normal[2](mu=0.0, sigma=1.0); "
            "IndependentCopula"
            ")"
        )
    )


def test_get_sample(joint_distribution) -> None:
    sample = joint_distribution.compute_samples(3)
    assert len(sample.shape) == 2
    assert sample.shape[0] == 3
    assert sample.shape[1] == 4


def test_get_cdf(joint_distribution) -> None:
    result = joint_distribution.compute_cdf(array([0] * 4))
    assert allclose(result, array([0.5] * 4))


def test_get_inverse_cdf(joint_distribution) -> None:
    result = joint_distribution.compute_inverse_cdf(array([0.5] * 4))
    assert allclose(result, array([0.0] * 4))


def test_cdf(joint_distribution) -> None:
    cdf = joint_distribution._cdf(1)
    assert cdf(0.0) == 0.5


def test_pdf(joint_distribution) -> None:
    pdf = joint_distribution._pdf(1)
    assert allclose(pdf(0.0), 0.398942, 1e-3)


def test_mean(joint_distribution) -> None:
    assert allclose(joint_distribution.mean, array([0.0] * 4))


def test_std(joint_distribution) -> None:
    assert allclose(joint_distribution.standard_deviation, array([1.0] * 4))


def test_support(joint_distribution) -> None:
    expectation = array([-inf, inf])
    for element in joint_distribution.support:
        assert allclose(element, expectation)


def test_range(joint_distribution) -> None:
    expectation = array([-7.650628, 7.650628])
    for element in joint_distribution.range:
        assert allclose(element, expectation, 1e-3)
