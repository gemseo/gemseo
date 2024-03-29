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

from gemseo.uncertainty.distributions.openturns.composed import OTComposedDistribution
from gemseo.uncertainty.distributions.openturns.normal import OTNormalDistribution

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.uncertainty.distributions.openturns.distribution import OTDistribution


@pytest.fixture(scope="module")
def distributions() -> list[OTNormalDistribution]:
    """Two normal distributions."""
    return [OTNormalDistribution(name, dimension=2) for name in ["x1", "x2"]]


@pytest.fixture(scope="module")
def composed_distribution(
    distributions: Sequence[OTDistribution],
) -> OTComposedDistribution:
    """A composed distribution."""
    return OTComposedDistribution(distributions)


@pytest.mark.parametrize("dimensions", [(1,), (1, 1), (2,)])
def test_repr(composed_distribution, dimensions) -> None:
    """Check the string representation of a composed distribution."""
    normal = "Normal(mu=0.0, sigma=1.0)"
    normal2 = "Normal[2](mu=0.0, sigma=1.0)"
    distributions = [OTNormalDistribution(dimension=dimensions[0])]
    if sum(dimensions) == 1:
        expected = normal
    elif len(dimensions) == 1:
        expected = f"OTComposedDistribution({normal2}; IndependentCopula)"
    else:
        expected = f"OTComposedDistribution({normal}, {normal}; IndependentCopula)"
        distributions.append(OTNormalDistribution(dimension=dimensions[1]))

    assert repr(OTComposedDistribution(distributions)) == expected


def test_constructor(composed_distribution) -> None:
    assert composed_distribution.dimension == 4
    assert composed_distribution.variable_name == "x1_x2"
    assert composed_distribution.distribution_name == "Composed"
    assert composed_distribution.transformation == "x1_x2"
    assert len(composed_distribution.parameters) == 1
    assert composed_distribution.parameters[0] is None


def test_copula(distributions) -> None:
    """Check the use of an OpenTURNS copula."""
    distribution = OTComposedDistribution(distributions, copula=NormalCopula(4))
    assert repr(distribution) == (
        "OTComposedDistribution("
        "Normal[2](mu=0.0, sigma=1.0), Normal[2](mu=0.0, sigma=1.0); NormalCopula)"
    )
    assert distribution.distribution.getCopula().getName() == "NormalCopula"


def test_variable_name(distributions) -> None:
    """Check the use of a custom variable name."""
    assert OTComposedDistribution(distributions, variable="foo").variable_name == "foo"


def test_str(composed_distribution) -> None:
    """Check the string representation of the composed distribution."""
    assert (
        repr(composed_distribution)
        == str(composed_distribution)
        == (
            "OTComposedDistribution("
            "Normal[2](mu=0.0, sigma=1.0), "
            "Normal[2](mu=0.0, sigma=1.0); "
            "IndependentCopula"
            ")"
        )
    )


def test_get_sample(composed_distribution) -> None:
    sample = composed_distribution.compute_samples(3)
    assert len(sample.shape) == 2
    assert sample.shape[0] == 3
    assert sample.shape[1] == 4


def test_get_cdf(composed_distribution) -> None:
    result = composed_distribution.compute_cdf(array([0] * 4))
    assert allclose(result, array([0.5] * 4))


def test_get_inverse_cdf(composed_distribution) -> None:
    result = composed_distribution.compute_inverse_cdf(array([0.5] * 4))
    assert allclose(result, array([0.0] * 4))


def test_cdf(composed_distribution) -> None:
    cdf = composed_distribution._cdf(1)
    assert cdf(0.0) == 0.5


def test_pdf(composed_distribution) -> None:
    pdf = composed_distribution._pdf(1)
    assert allclose(pdf(0.0), 0.398942, 1e-3)


def test_mean(composed_distribution) -> None:
    assert allclose(composed_distribution.mean, array([0.0] * 4))


def test_std(composed_distribution) -> None:
    assert allclose(composed_distribution.standard_deviation, array([1.0] * 4))


def test_support(composed_distribution) -> None:
    expectation = array([-inf, inf])
    for element in composed_distribution.support:
        assert allclose(element, expectation)


def test_range(composed_distribution) -> None:
    expectation = array([-7.650628, 7.650628])
    for element in composed_distribution.range:
        assert allclose(element, expectation, 1e-3)
