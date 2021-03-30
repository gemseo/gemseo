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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

from __future__ import absolute_import, division, unicode_literals

import numbers

import pytest
from future import standard_library
from numpy import allclose, array, inf, vstack
from numpy.random import exponential, lognormal, normal, rand, seed, weibull

from gemseo.core.dataset import Dataset
from gemseo.uncertainty.statistics.parametric import ParametricStatistics

standard_library.install_aliases()


@pytest.fixture(scope="module")
def random_sample():
    """This fixture is a random sample of four random variables
    distributed according to the uniform, normal, weibull and exponential
    probability distributions."""
    seed(0)
    n_samples = 100
    uniform_rand = rand(n_samples)
    normal_rand = normal(size=n_samples)
    weibull_rand = weibull(1.5, size=n_samples)
    exponential_rand = exponential(size=n_samples)
    data = vstack((uniform_rand, normal_rand, weibull_rand, exponential_rand)).T
    dataset = Dataset()
    dataset.set_from_array(data, ["X_0", "X_1", "X_2", "X_3"])
    theoretical_distributions = {
        "X_0": "Uniform",
        "X_1": "Normal",
        "X_2": "WeibullMin",
        "X_3": "Exponential",
    }
    tested_distributions = ["Exponential", "Normal", "Uniform"]
    return (dataset, tested_distributions, theoretical_distributions)


def test_distfitstats_constructor(random_sample):
    """ Test constructor """
    dataset, tested_distributions, _ = random_sample
    ParametricStatistics(dataset, tested_distributions)
    with pytest.raises(ValueError):
        ParametricStatistics(dataset, tested_distributions, fitting_criterion="dummy")


def test_distfitstats_str(random_sample):
    """ Test constructor """
    dataset, tested_distributions, _ = random_sample
    stat = ParametricStatistics(dataset, tested_distributions)
    assert "ParametricStatistics" in str(stat)


def test_distfitstats_properties(random_sample):
    """ Test standard properties """
    dataset, tested_distributions, _ = random_sample
    stats = ParametricStatistics(dataset, tested_distributions)
    assert stats.n_samples == dataset.n_samples
    assert stats.n_variables == dataset.n_variables


def test_distfitstats_getcrit(random_sample):
    """ Test methods relative to criteria """
    dataset, tested_distributions, _ = random_sample
    stats = ParametricStatistics(dataset, tested_distributions)
    criteria, is_pvalue = stats.get_criteria("X_0")
    assert is_pvalue == False
    for distribution, criterion in criteria.items():
        assert distribution in tested_distributions
        assert isinstance(criterion, numbers.Number)
    stats = ParametricStatistics(
        dataset, tested_distributions, fitting_criterion="Kolmogorov"
    )
    criteria, is_pvalue = stats.get_criteria("X_0")
    assert is_pvalue == True
    for distribution, criterion in criteria.items():
        assert distribution in tested_distributions
        assert isinstance(criterion, numbers.Number)
    with pytest.raises(ValueError):
        stats = ParametricStatistics(dataset, ["dummy"])
    stats = ParametricStatistics(dataset, ["Normal"], fitting_criterion="Kolmogorov")
    stats = ParametricStatistics(
        dataset, ["Normal"], fitting_criterion="Kolmogorov", selection_criterion="first"
    )


def test_distfitstats_statistics(random_sample):
    """ Test standard statistics """
    dataset, tested_distributions, _ = random_sample
    stats = ParametricStatistics(dataset, tested_distributions)
    stats.maximum()
    stats.mean()
    stats.minimum()
    stats.range()
    thresh = {name: array([0.0]) for name in ["X_0", "X_1", "X_2", "X_3"]}
    stats.probability(thresh, greater=True)
    stats.probability(thresh, greater=False)
    stats.moment(1)
    stats.variance()
    stats.standard_deviation()
    stats.quantile(0.5)


def test_distfitstats_plot(random_sample, tmpdir):
    """ Test plot methods """
    array, tested_distributions, _ = random_sample
    directory = str(tmpdir.mkdir("plot"))
    stats = ParametricStatistics(array, tested_distributions)
    stats.plot_criteria("X_1", save=True, show=False, directory=directory)
    stats.plot_criteria(
        "X_1", save=True, show=False, directory=directory, title="title"
    )
    with pytest.raises(ValueError):
        stats.plot_criteria("dummy", save=True, show=False, directory=directory)
    stats = ParametricStatistics(
        array, tested_distributions, fitting_criterion="Kolmogorov"
    )
    stats.plot_criteria("X_1", save=True, show=False, directory=directory)


def test_distfitstats_tolint(random_sample):
    """ Test tolerance_interval() method """
    dataset, tested_distributions, _ = random_sample
    stats = ParametricStatistics(dataset, tested_distributions)
    with pytest.raises(ValueError):
        stats.tolerance_interval(1.5)
    with pytest.raises(ValueError):
        stats.tolerance_interval(0.1, confidence=-1.6)
    with pytest.raises(ValueError):
        stats.tolerance_interval(0.1, side="dummy")
    stats = ParametricStatistics(dataset, ["ChiSquare"])
    with pytest.raises(ValueError):
        stats.tolerance_interval(0.1)
    for dist in ["Normal", "Uniform", "LogNormal", "WeibullMin", "Exponential"]:
        stats = ParametricStatistics(dataset, [dist])
        stats.tolerance_interval(0.1)
        stats.tolerance_interval(0.1, side="both")
        stats.tolerance_interval(0.1, side="upper")
        stats.tolerance_interval(0.1, side="lower")


def test_distfitstats_tolint_normal():
    """Test returned values by tolerance_interval() method
    for Normal distribution"""
    seed(0)
    n_samples = 100
    normal_rand = normal(size=n_samples).reshape((-1, 1))
    dataset = Dataset()
    dataset.set_from_array(normal_rand)
    stats = ParametricStatistics(dataset, ["Normal"])
    limits = stats.tolerance_interval(0.1, side="both")
    assert allclose(limits["x_0"][0], array([-0.085272]), rtol=1e-5)
    assert allclose(limits["x_0"][1], array([0.204888]), rtol=1e-5)
    limits = stats.tolerance_interval(0.1, side="upper")
    assert allclose(limits["x_0"][0], array([-inf]), rtol=1e-5)
    assert allclose(limits["x_0"][1], array([-1.070161]), rtol=1e-5)
    limits = stats.tolerance_interval(0.1, side="lower")
    assert allclose(limits["x_0"][0], array([1.189777]), rtol=1e-5)
    assert allclose(limits["x_0"][1], array([inf]), rtol=1e-5)


def test_distfitstats_tolint_uniform():
    """Test returned values by tolerance_interval() method
    for Uniform distribution"""
    seed(0)
    n_samples = 100
    uniform_rand = rand(n_samples).reshape((-1, 1))
    dataset = Dataset()
    dataset.set_from_array(uniform_rand)
    stats = ParametricStatistics(dataset, ["Uniform"])
    limits = stats.tolerance_interval(0.1, side="both")
    assert allclose(limits["x_0"][0], array([0.446501]), rtol=1e-5)
    assert allclose(limits["x_0"][1], array([0.567412]), rtol=1e-5)
    limits = stats.tolerance_interval(0.1, side="upper")
    assert allclose(limits["x_0"][0], array([-inf]), rtol=1e-5)
    assert allclose(limits["x_0"][1], array([0.098398]), rtol=1e-5)
    limits = stats.tolerance_interval(0.1, side="lower")
    assert allclose(limits["x_0"][0], array([0.898184]), rtol=1e-5)
    assert allclose(limits["x_0"][1], array([inf]), rtol=1e-5)


def test_distfitstats_tolint_lognormal():
    """Test returned values by tolerance_interval() method
    for Lognormal distribution"""
    seed(0)
    n_samples = 100
    lognormal_rand = lognormal(size=n_samples).reshape((-1, 1))
    dataset = Dataset()
    dataset.set_from_array(lognormal_rand)
    stats = ParametricStatistics(dataset, ["LogNormal"])
    limits = stats.tolerance_interval(0.1, side="both")
    assert allclose(limits["x_0"][0], array([0.884838]), rtol=1e-5)
    assert allclose(limits["x_0"][1], array([1.192188]), rtol=1e-5)
    limits = stats.tolerance_interval(0.1, side="upper")
    assert allclose(limits["x_0"][0], array([0.0]), rtol=1e-5)
    assert allclose(limits["x_0"][1], array([0.321639]), rtol=1e-5)
    limits = stats.tolerance_interval(0.1, side="lower")
    assert allclose(limits["x_0"][0], array([3.279741]), rtol=1e-5)
    assert allclose(limits["x_0"][1], array([inf]), rtol=1e-5)


def test_distfitstats_tolint_weibull(random_sample):
    """Test returned values by tolerance_interval() method
    for Weibull distribution"""
    seed(0)
    n_samples = 100
    import openturns as ot

    weibull_rand = array(ot.WeibullMin().getSample(n_samples)).reshape((-1, 1))
    dataset = Dataset()
    dataset.set_from_array(weibull_rand)
    stats = ParametricStatistics(dataset, ["WeibullMin"])
    limits = stats.tolerance_interval(0.1, side="both")
    assert allclose(limits["x_0"][0], array([0.8]), atol=1e-1)
    assert allclose(limits["x_0"][1], array([0.7]), atol=1e-1)
    limits = stats.tolerance_interval(0.1, side="upper")
    assert allclose(limits["x_0"][0], array([0.0]), atol=1e-1)
    assert allclose(limits["x_0"][1], array([0.1]), atol=1e-1)
    limits = stats.tolerance_interval(0.1, side="lower")
    assert allclose(limits["x_0"][0], array([2.5]), atol=1e-1)
    assert allclose(limits["x_0"][1], array([inf]), atol=1e-51)


def test_distfitstats_tolint_exponential(random_sample):
    """Test returned values by tolerance_interval() method
    for Exponential distribution"""
    seed(0)
    n_samples = 100
    exp_rand = exponential(size=n_samples).reshape((-1, 1))
    dataset = Dataset()
    dataset.set_from_array(exp_rand)
    stats = ParametricStatistics(dataset, ["Exponential"])
    limits = stats.tolerance_interval(0.1, side="both")
    assert allclose(limits["x_0"][0], array([0.547349]), rtol=1e-5)
    assert allclose(limits["x_0"][1], array([0.729509]), rtol=1e-5)
    limits = stats.tolerance_interval(0.1, side="upper")
    assert allclose(limits["x_0"][0], array([0.00466]), rtol=1e-4)
    assert allclose(limits["x_0"][1], array([2.157938]), rtol=1e-5)
    limits = stats.tolerance_interval(0.1, side="lower")
    assert allclose(limits["x_0"][0], array([0.103189]), rtol=1e-5)
    assert allclose(limits["x_0"][1], array([inf]), rtol=1e-5)


def test_distfitstats_abvalue_normal():
    """ Test """
    seed(0)
    n_samples = 100
    normal_rand = normal(size=n_samples).reshape((-1, 1))
    dataset = Dataset()
    dataset.set_from_array(normal_rand)
    stats = ParametricStatistics(dataset, ["Normal"])
    assert allclose(stats.a_value()["x_0"], array([-1.477877]), rtol=1e-5)
    assert allclose(stats.b_value()["x_0"], array([-1.406543]), rtol=1e-5)


def test_distfitstats_available(random_sample):
    dataset, tested_distributions, _ = random_sample
    stat = ParametricStatistics(dataset, tested_distributions)
    assert "Normal" in ParametricStatistics.get_available_distributions()
    assert "BIC" in ParametricStatistics.get_available_criteria()
    assert "Kolmogorov" in ParametricStatistics.get_significance_tests()
    assert "Normal" in stat.get_fitting_matrix()
