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
from gemseo.uncertainty.distributions.openturns.exponential import (
    OTExponentialDistribution,
)
from gemseo.uncertainty.distributions.openturns.fitting import OTDistributionFitter
from gemseo.uncertainty.distributions.openturns.normal import OTNormalDistribution
from gemseo.uncertainty.distributions.openturns.triangular import (
    OTTriangularDistribution,
)
from gemseo.uncertainty.distributions.openturns.uniform import OTUniformDistribution
from numpy import allclose
from numpy import array
from numpy import inf
from numpy import ndarray
from numpy.random import randn
from numpy.random import seed
from openturns import RandomGenerator


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


def test_compute_samples():
    RandomGenerator.SetSeed(0)
    distribution = OTDistribution("x", "Normal", (0, 2))
    sample = distribution.compute_samples(3)
    assert isinstance(sample, ndarray)
    assert len(sample.shape) == 2
    assert sample.shape[0] == 3
    assert sample.shape[1] == 1
    expectation = array([[1.216403], [-2.532346], [-0.876531]])
    assert allclose(sample, expectation, 1e-3)
    distribution = OTDistribution("x", "Normal", (0, 2), 4)
    sample = distribution.compute_samples(3)
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
    result = distribution.compute_cdf(array([0, 0]))
    assert allclose(result, array([0.5, 0.5]))


def test_get_inverse_cdf():
    distribution = OTDistribution("x", "Normal", (0, 2), 2)
    result = distribution.compute_inverse_cdf(array([0.5, 0.5]))
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
    distribution = OTDistribution(
        "x", "Normal", (0, 2), 2, lower_bound=0.0, upper_bound=1.0
    )
    expectation = array([0.0, 1.0])
    for element in distribution.support:
        assert allclose(element, expectation, 1e-3)

    distribution = OTDistribution("x", "Uniform", (0, 1), 2, lower_bound=0.5)
    expectation = array([0.5, 1.0])
    for element in distribution.support:
        assert allclose(element, expectation, 1e-3)

    distribution = OTDistribution("x", "Uniform", (0, 1), 2, upper_bound=0.5)
    expectation = array([0.0, 0.5])
    for element in distribution.support:
        assert allclose(element, expectation, 1e-3)

    with pytest.raises(ValueError):
        OTDistribution("x", "Uniform", (0, 1), 2, upper_bound=1.5)

    with pytest.raises(ValueError):
        OTDistribution("x", "Uniform", (0, 1), 2, lower_bound=-0.5)

    with pytest.raises(ValueError):
        OTDistribution("x", "Uniform", (0, 1), 2, lower_bound=-0.5, upper_bound=1.0)

    with pytest.raises(ValueError):
        OTDistribution("x", "Uniform", (0, 1), 2, lower_bound=0.0, upper_bound=1.5)


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


def test_plot(tmp_wd):
    distribution = OTTriangularDistribution("x", dimension=2)
    figures = distribution.plot_all(False, True, directory_path=tmp_wd)
    assert (tmp_wd / "distribution_x_0.png").exists()
    assert (tmp_wd / "distribution_x_1.png").exists()
    assert len(figures) == 2


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
        factory.compute_measure(dist, "BIC")


def test_otdistfitter_criterion(norm_data):
    factory = OTDistributionFitter("x", norm_data)
    with pytest.raises(ValueError):
        factory.compute_measure("Normal", "Dummy")


def test_otdistfitter_fit(norm_data):
    factory = OTDistributionFitter("x", norm_data)
    dist = factory.fit("Normal")
    assert isinstance(dist, OTDistribution)


def tst_otdistfitter_bic(norm_data):
    factory = OTDistributionFitter("x", norm_data)
    dist = factory.fit("Normal")
    quality_measure = factory.compute_measure(dist, "BIC")
    assert allclose(quality_measure, 2.59394512877)
    factory = OTDistributionFitter("x", norm_data)
    quality_measure = factory.compute_measure("Normal", "BIC")
    assert allclose(quality_measure, 2.59394512877)


def test_otdistfitter_kolmogorov(norm_data):
    factory = OTDistributionFitter("x", norm_data)
    dist = factory.fit("Normal")
    acceptable, details = factory.compute_measure(dist, "Kolmogorov")
    assert acceptable
    assert "statistics" in details
    assert "p-value" in details
    assert "level" in details
    assert details["level"] == 0.05
    assert allclose(details["statistics"], 0.04330972976650932)
    assert allclose(details["p-value"], 0.9879299613543082)


def test_otdistfitter_available(norm_data):
    factory = OTDistributionFitter("x", norm_data)
    assert "BIC" in factory.available_criteria
    assert "BIC" not in factory.available_significance_tests
    assert "Normal" in factory.available_distributions


def test_otdistfitter_select(norm_data):
    factory = OTDistributionFitter("x", norm_data)
    dist = factory.select(["Normal", "Exponential"], "BIC")
    assert isinstance(dist, OTDistribution)
