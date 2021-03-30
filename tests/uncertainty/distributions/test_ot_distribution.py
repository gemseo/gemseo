# -*- coding: utf-8 -*-
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
from __future__ import absolute_import, division, unicode_literals

from os import remove
from os.path import exists

import pytest
from future import standard_library
from numpy import allclose, array, inf, ndarray
from numpy.random import randn, seed
from openturns import RandomGenerator

from gemseo.uncertainty.distributions.ot_dist import (
    OTDistribution,
    OTExponentialDistribution,
    OTNormalDistribution,
    OTTriangularDistribution,
    OTUniformDistribution,
)
from gemseo.uncertainty.distributions.ot_fdist import OTDistributionFitter

standard_library.install_aliases()


def test_constructor():
    distribution = OTDistribution("x", "Normal", (0, 1))
    assert distribution.dimension == 1
    assert distribution.variable_name == "x"
    assert distribution.distribution_name == "Normal"
    assert distribution.transformation == "x"
    assert len(distribution.parameters) == 2
    assert distribution.parameters[0] == 0
    assert distribution.parameters[1] == 1


def test_bad_distribution():
    with pytest.raises(ValueError):
        OTDistribution("x", "Dummy", (0, 1))


def test_bad_distribution_parameters():
    with pytest.raises(ValueError):
        OTDistribution("x", "Normal", (0, 1, 2))


def test_str():
    distribution = OTDistribution("x", "Normal", (0, 2))
    assert str(distribution) == "Normal(0, 2)"
    distribution = OTDistribution(
        "x", "Normal", (0, 2), standard_parameters={"mean": 0, "var": 4}
    )
    assert str(distribution) == "Normal(mean=0, var=4)"


def test_get_sample():
    RandomGenerator.SetSeed(0)
    distribution = OTDistribution("x", "Normal", (0, 2))
    sample = distribution.get_sample(3)
    assert isinstance(sample, ndarray)
    assert len(sample.shape) == 2
    assert sample.shape[0] == 3
    assert sample.shape[1] == 1
    expectation = array([[1.216403], [-2.532346], [-0.876531]])
    assert allclose(sample, expectation, 1e-3)
    distribution = OTDistribution("x", "Normal", (0, 2), 4)
    sample = distribution.get_sample(3)
    expectation = array(
        [
            [2.410956, -0.710014, 1.586312, -4.580124],
            [-4.36277, 2.874499, -0.941051, -2.565771],
            [0.700084, 1.621336, 0.522036, -2.623562],
        ]
    )
    assert allclose(sample, expectation, 1e-3)


def test_get_cdf():
    distribution = OTDistribution("x", "Normal", (0, 2), 2)
    result = distribution.cdf(array([0, 0]))
    assert allclose(result, array([0.5, 0.5]))


def test_get_inverse_cdf():
    distribution = OTDistribution("x", "Normal", (0, 2), 2)
    result = distribution.inverse_cdf(array([0.5, 0.5]))
    assert allclose(result, array([0.0, 0.0]))


def test_cdf():
    distribution = OTDistribution("x", "Normal", (0, 2), 2)
    cdf = distribution._cdf(1)
    assert cdf(0.0) == 0.5


def test_pdf():
    distribution = OTDistribution("x", "Normal", (0, 2), 2)
    pdf = distribution._pdf(1)
    assert allclose(pdf(0.0), 0.19947114020071632, 1e-3)


def test_mean():
    distribution = OTDistribution("x", "Normal", (0, 2), 2)
    assert allclose(distribution.mean, array([0.0, 0.0]))


def test_std():
    distribution = OTDistribution("x", "Normal", (0, 2), 2)
    assert allclose(distribution.standard_deviation, array([2.0, 2.0]))


def test_support():
    distribution = OTDistribution("x", "Normal", (0, 2), 2)
    expectation = array([-inf, inf])
    for element in distribution.support:
        assert allclose(element, expectation)


def test_range():
    distribution = OTDistribution("x", "Normal", (0, 2), 2)
    expectation = array([-15.301256, 15.301256])
    for element in distribution.range:
        assert allclose(element, expectation, 1e-3)
    distribution = OTDistribution("x", "Uniform", (0, 1), 2)
    expectation = array([0.0, 1.0])
    for element in distribution.range:
        assert allclose(element, expectation, 1e-3)


def test_truncation():
    distribution = OTDistribution("x", "Normal", (0, 2), 2, l_b=0.0, u_b=1.0)
    expectation = array([0.0, 1.0])
    for element in distribution.support:
        assert allclose(element, expectation, 1e-3)

    distribution = OTDistribution("x", "Uniform", (0, 1), 2, l_b=0.5)
    expectation = array([0.5, 1.0])
    for element in distribution.support:
        assert allclose(element, expectation, 1e-3)

    distribution = OTDistribution("x", "Uniform", (0, 1), 2, u_b=0.5)
    expectation = array([0.0, 0.5])
    for element in distribution.support:
        assert allclose(element, expectation, 1e-3)

    with pytest.raises(ValueError):
        OTDistribution("x", "Uniform", (0, 1), 2, u_b=1.5)

    with pytest.raises(ValueError):
        OTDistribution("x", "Uniform", (0, 1), 2, l_b=-0.5)

    with pytest.raises(ValueError):
        OTDistribution("x", "Uniform", (0, 1), 2, l_b=-0.5, u_b=1.0)

    with pytest.raises(ValueError):
        OTDistribution("x", "Uniform", (0, 1), 2, l_b=0.0, u_b=1.5)


def test_transformation():
    distribution = OTDistribution("x", "Uniform", (0, 2), 2, transformation="2*x")
    expectation = array([0.0, 4.0])
    for element in distribution.support:
        assert allclose(element, expectation, atol=1e-3)


def test_normal():
    distribution = OTNormalDistribution("x")
    assert str(distribution) == "Normal(mu=0.0, sigma=1.0)"


def test_uniform():
    distribution = OTUniformDistribution("x")
    assert str(distribution) == "Uniform(lower=0.0, upper=1.0)"


def test_exponential():
    distribution = OTExponentialDistribution("x")
    assert str(distribution) == "Exponential(loc=0.0, rate=1.0)"


def test_triangular():
    distribution = OTTriangularDistribution("x")
    assert str(distribution) == "Triangular(lower=0.0, mode=0.5, upper=1.0)"


def test_plot():
    distribution = OTTriangularDistribution("x", dimension=2)
    distribution.plot_all(False, True)
    assert exists("distribution_x_0.pdf")
    assert exists("distribution_x_1.pdf")
    remove("distribution_x_0.pdf")
    remove("distribution_x_1.pdf")
    distribution.plot_all(False, True, "prefix")
    assert exists("prefix_distribution_x_0.pdf")
    assert exists("prefix_distribution_x_1.pdf")
    remove("prefix_distribution_x_0.pdf")
    remove("prefix_distribution_x_1.pdf")


@pytest.fixture
def norm_data():
    seed(1)
    return randn(100)


def test_otdistfitter_constructor():
    with pytest.raises(TypeError):
        OTDistributionFitter("x", {"x_" + str(index): index for index in range(100)})


def test_otdistfitter_distribution(norm_data):
    factory = OTDistributionFitter("x", norm_data)
    with pytest.raises(ValueError):
        factory.fit("Dummy")
    with pytest.raises(TypeError):
        dist = OTNormalDistribution("x", dimension=2)
        factory.measure(dist, "BIC")


def test_otdistfitter_criterion(norm_data):
    factory = OTDistributionFitter("x", norm_data)
    with pytest.raises(ValueError):
        factory.measure("Normal", "Dummy")


def test_otdistfitter_fit(norm_data):
    factory = OTDistributionFitter("x", norm_data)
    dist = factory.fit("Normal")
    assert isinstance(dist, OTDistribution)


def test_otdistfitter_BIC(norm_data):
    factory = OTDistributionFitter("x", norm_data)
    dist = factory.fit("Normal")
    quality_measure = factory.measure(dist, "BIC")
    assert allclose(quality_measure, 2.59394512877)
    factory = OTDistributionFitter("x", norm_data)
    quality_measure = factory.measure("Normal", "BIC")
    assert allclose(quality_measure, 2.59394512877)


def test_otdistfitter_kolmogorov(norm_data):
    factory = OTDistributionFitter("x", norm_data)
    dist = factory.fit("Normal")
    acceptable, details = factory.measure(dist, "Kolmogorov")
    assert acceptable
    assert "statistics" in details
    assert "p-value" in details
    assert "level" in details
    assert details["level"] == 0.05
    assert allclose(details["statistics"], 0.04330972976650932)
    assert allclose(details["p-value"], 0.9879299613543082)


def test_otdistfitter_get(norm_data):
    factory = OTDistributionFitter("x", norm_data)
    assert "BIC" in factory.get_available_criteria()
    assert "BIC" not in factory.get_significance_tests()
    assert "Normal" in factory.get_available_distributions()


def test_otdistfitter_select(norm_data):
    factory = OTDistributionFitter("x", norm_data)
    dist = factory.select(["Normal", "Exponential"], "BIC")
    assert isinstance(dist, OTDistribution)
