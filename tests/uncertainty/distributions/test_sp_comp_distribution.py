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

from typing import Sequence

import pytest
from gemseo.uncertainty.distributions.scipy.composed import SPComposedDistribution
from gemseo.uncertainty.distributions.scipy.distribution import SPDistribution
from gemseo.uncertainty.distributions.scipy.normal import SPNormalDistribution
from numpy import allclose
from numpy import array
from numpy import inf
from numpy.random import seed


@pytest.fixture(scope="module")
def distributions() -> list[SPNormalDistribution]:
    """Two normal distributions."""
    return [SPNormalDistribution(name, dimension=2) for name in ["x1", "x2"]]


@pytest.fixture(scope="module")
def composed_distribution(
    distributions: Sequence[SPDistribution],
) -> SPComposedDistribution:
    """The composed distribution."""
    return SPComposedDistribution(distributions)


def test_available_copula_models():
    """Check AVAILABLE_COPULA_MODELS."""
    assert "independent_copula" in SPComposedDistribution.AVAILABLE_COPULA_MODELS


def test_constructor(composed_distribution):
    assert composed_distribution.dimension == 4
    assert composed_distribution.variable_name == "x1_x2"
    assert composed_distribution.distribution_name == "Composed"
    assert composed_distribution.transformation == "x1_x2"
    assert len(composed_distribution.parameters) == 1
    assert composed_distribution.parameters[0].name == "independent_copula"


@pytest.mark.parametrize(
    "copula",
    [SPComposedDistribution.CopulaModel.independent_copula, "independent_copula"],
)
def test_copula_enum_or_str(distributions, copula):
    """Check that copula passed to __init__ can be either a CopulaModel or a str."""
    assert (
        str(SPComposedDistribution(distributions, copula=copula).parameters[0])
        == "independent_copula"
    )


def test_variable_name(distributions):
    """Check the use of a custom variable name."""
    assert SPComposedDistribution(distributions, variable="foo").variable_name == "foo"


def test_str(composed_distribution):
    assert str(composed_distribution) == "Composed(independent_copula)"


def test_compute_samples(composed_distribution):
    seed(0)
    sample = composed_distribution.compute_samples(3)
    assert len(sample.shape) == 2
    assert sample.shape[0] == 3
    assert sample.shape[1] == 4


def test_get_cdf(composed_distribution):
    result = composed_distribution.compute_cdf(array([0] * 4))
    assert allclose(result, array([0.5] * 4))


def test_get_inverse_cdf(composed_distribution):
    result = composed_distribution.compute_inverse_cdf(array([0.5] * 4))
    assert allclose(result, array([0.0] * 4))


def test_cdf(composed_distribution):
    cdf = composed_distribution._cdf(1)
    assert cdf(0.0) == 0.5


def test_pdf(composed_distribution):
    pdf = composed_distribution._pdf(1)
    assert allclose(pdf(0.0), 0.398942, 1e-3)


def test_mean(composed_distribution):
    assert allclose(composed_distribution.mean, array([0.0] * 4))


def test_std(composed_distribution):
    assert allclose(composed_distribution.standard_deviation, array([1.0] * 4))


def test_support(composed_distribution):
    expectation = array([-inf, inf])
    for element in composed_distribution.support:
        assert allclose(element, expectation)


def test_range(composed_distribution):
    expectation = array([-7.034484, 7.034487])
    for element in composed_distribution.range:
        assert allclose(element, expectation, 1e-3)
