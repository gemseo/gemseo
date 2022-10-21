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
from gemseo.uncertainty.distributions.openturns.distribution import OTDistribution
from gemseo.uncertainty.distributions.openturns.normal import OTNormalDistribution
from numpy import allclose
from numpy import array
from numpy import inf
from numpy.random import seed


@pytest.fixture
def composed_distribution():
    distribution1 = OTNormalDistribution("x1", dimension=2)
    distribution2 = OTNormalDistribution("x2", dimension=2)
    distributions = [distribution1, distribution2]
    return OTDistribution._COMPOSED_DISTRIBUTION(distributions)


def test_constructor(composed_distribution):
    assert composed_distribution.dimension == 4
    assert composed_distribution.variable_name == "x1_x2"
    assert composed_distribution.distribution_name == "Composed"
    assert composed_distribution.transformation == "x1_x2"
    assert len(composed_distribution.parameters) == 1
    assert composed_distribution.parameters[0] == "independent_copula"


def test_str(composed_distribution):
    assert str(composed_distribution) == "Composed(independent_copula)"


def test_get_sample(composed_distribution):
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
    expectation = array([-7.650628, 7.650628])
    for element in composed_distribution.range:
        assert allclose(element, expectation, 1e-3)
