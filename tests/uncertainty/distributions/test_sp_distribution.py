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
from gemseo.uncertainty.distributions.scipy.composed import SPComposedDistribution
from gemseo.uncertainty.distributions.scipy.distribution import SPDistribution
from gemseo.uncertainty.distributions.scipy.exponential import SPExponentialDistribution
from gemseo.uncertainty.distributions.scipy.normal import SPNormalDistribution
from gemseo.uncertainty.distributions.scipy.triangular import SPTriangularDistribution
from gemseo.uncertainty.distributions.scipy.uniform import SPUniformDistribution
from numpy import allclose
from numpy import array
from numpy import inf
from numpy import ndarray
from numpy.random import seed


def test_composed_distribution():
    """Check the composed distribution associated with a SPDistribution."""
    assert SPDistribution._COMPOSED_DISTRIBUTION == SPComposedDistribution


def test_constructor():
    distribution = SPDistribution("x", "norm", {"loc": 0.0, "scale": 1})
    assert distribution.dimension == 1
    assert distribution.variable_name == "x"
    assert distribution.distribution_name == "norm"
    assert distribution.transformation == "x"
    assert len(distribution.parameters) == 2
    assert distribution.parameters["loc"] == 0
    assert distribution.parameters["scale"] == 1


def test_bad_distribution():
    with pytest.raises(ValueError):
        SPDistribution("x", "Dummy", {"loc": 0.0, "scale": 1})


def test_bad_distribution_parameters():
    with pytest.raises(ValueError):
        SPDistribution("x", "norm", {"loc": 0.0, "max": 1})


def test_str():
    distribution = SPDistribution("x", "norm", {"loc": 0.0, "scale": 2.0})
    assert str(distribution) == "norm(loc=0.0, scale=2.0)"
    distribution = SPDistribution(
        "x",
        "norm",
        {"loc": 0.0, "scale": 2.0},
        standard_parameters={"mean": 0, "var": 4},
    )
    assert str(distribution) == "norm(mean=0, var=4)"


def test_compute_samples():
    seed(0)
    distribution = SPDistribution("x", "norm", {"loc": 0.0, "scale": 2})
    sample = distribution.compute_samples(3)
    assert isinstance(sample, ndarray)
    assert len(sample.shape) == 2
    assert sample.shape[0] == 3
    assert sample.shape[1] == 1
    expectation = array([[3.528105], [0.800314], [1.957476]])
    assert allclose(sample, expectation, 1e-3)
    distribution = SPDistribution("x", "norm", {"loc": 0.0, "scale": 2}, 4)
    sample = distribution.compute_samples(3)
    expectation = array(
        [
            [4.481786, 1.900177, 0.821197, 1.522075],
            [3.735116, -0.302714, 0.288087, 0.24335],
            [-1.954556, -0.206438, 2.908547, 0.887726],
        ]
    )
    assert allclose(sample, expectation, 1e-3)


def test_get_cdf():
    distribution = SPDistribution("x", "norm", {"loc": 0.0, "scale": 2}, 2)
    result = distribution.compute_cdf(array([0, 0]))
    assert allclose(result, array([0.5, 0.5]))


def test_get_inverse_cdf():
    distribution = SPDistribution("x", "norm", {"loc": 0.0, "scale": 2}, 2)
    result = distribution.compute_inverse_cdf(array([0.5, 0.5]))
    assert allclose(result, array([0.0, 0.0]))


def test_cdf():
    distribution = SPDistribution("x", "norm", {"loc": 0.0, "scale": 2}, 2)
    cdf = distribution._cdf(1)
    assert cdf(0.0) == 0.5


def test_pdf():
    distribution = SPDistribution("x", "norm", {"loc": 0.0, "scale": 2}, 2)
    pdf = distribution._pdf(1)
    assert allclose(pdf(0.0), 0.19947114020071632, 1e-3)


def test_mean():
    distribution = SPDistribution("x", "norm", {"loc": 0.0, "scale": 2}, 2)
    assert allclose(distribution.mean, array([0.0, 0.0]))


def test_std():
    distribution = SPDistribution("x", "norm", {"loc": 0.0, "scale": 2}, 2)
    assert allclose(distribution.standard_deviation, array([2.0, 2.0]))


def test_support():
    distribution = SPDistribution("x", "norm", {"loc": 0.0, "scale": 2}, 2)
    expectation = array([-inf, inf])
    for element in distribution.support:
        assert allclose(element, expectation)


def test_range():
    distribution = SPDistribution("x", "norm", {"loc": 0.0, "scale": 2}, 2)
    expectation = array([-14.068968, 14.068974])
    for element in distribution.range:
        assert allclose(element, expectation, 1e-3)


def test_normal():
    distribution = SPNormalDistribution("x")
    assert str(distribution) == "norm(mu=0.0, sigma=1.0)"


def test_uniform():
    distribution = SPUniformDistribution("x")
    assert str(distribution) == "uniform(lower=0.0, upper=1.0)"


def test_exponential():
    distribution = SPExponentialDistribution("x")
    assert str(distribution) == "expon(loc=0.0, scale=1.0)"


def test_triangular():
    distribution = SPTriangularDistribution("x")
    assert str(distribution) == "triang(lower=0.0, mode=0.5, upper=1.0)"
