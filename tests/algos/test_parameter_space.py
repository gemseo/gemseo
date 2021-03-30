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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

from __future__ import absolute_import, division, unicode_literals

import pytest
from future import standard_library
from numpy import allclose, array, ndarray

from gemseo.algos.parameter_space import ParameterSpace
from gemseo.uncertainty.distributions.distribution import ComposedDistribution

standard_library.install_aliases()


def test_constructor():
    space = ParameterSpace()
    assert not space.is_deterministic("x")
    assert not space.is_uncertain("x")


def test_add_variable():
    space = ParameterSpace()
    space.add_variable("x")
    assert space.is_deterministic("x")
    assert not space.is_uncertain("x")
    assert "x" not in space.marginals


def test_add_random_variable():
    space = ParameterSpace()
    space.add_variable("x")
    space.add_random_variable("y", "SPNormalDistribution", mu=0.0, sigma=1.0)
    assert not space.is_deterministic("y")
    assert space.is_uncertain("y")
    assert space.variables_names == ["x", "y"]
    assert space.uncertain_variables == ["y"]
    assert "y" in space.marginals


def test_remove_variable():
    space = ParameterSpace()
    space.add_variable("x1")
    space.add_variable("x2")
    space.add_random_variable("y1", "SPNormalDistribution", mu=0.0, sigma=1.0)
    space.add_random_variable("y2", "SPNormalDistribution", mu=0.0, sigma=1.0)
    space.remove_variable("x2")
    assert space.variables_names == ["x1", "y1", "y2"]
    assert space.uncertain_variables == ["y1", "y2"]
    space.remove_variable("y1")
    assert space.variables_names == ["x1", "y2"]
    assert space.uncertain_variables == ["y2"]
    assert "y1" not in space.marginals


def test_copula():
    space = ParameterSpace(copula=ComposedDistribution.INDEPENDENT_COPULA)
    space.add_variable("x")
    space.add_random_variable("y", "SPNormalDistribution", mu=0.0, sigma=1.0)
    space.add_random_variable("z", "SPUniformDistribution", lower=0.0, upper=1.0)
    with pytest.raises(ValueError):
        space = ParameterSpace(copula="dummy")


def test_get_composed_distribution():
    space = ParameterSpace(copula=ComposedDistribution.INDEPENDENT_COPULA)
    space.add_random_variable("x", "SPNormalDistribution", mu=0.0, sigma=1.0, size=2)
    space.get_composed_distribution("x")


def test_get_marginal_distributions():
    space = ParameterSpace(copula=ComposedDistribution.INDEPENDENT_COPULA)
    space.add_random_variable("x", "SPNormalDistribution", mu=0.0, sigma=1.0, size=2)
    space.get_marginal_distributions("x")


def test_get_sample():
    space = ParameterSpace()
    space.add_variable("x1")
    space.add_variable("x2")
    space.add_random_variable("y1", "SPNormalDistribution", mu=0.0, sigma=1.0)
    space.add_random_variable("y2", "SPNormalDistribution", mu=0.0, sigma=1.0, size=3)
    sample = space.get_sample(2, False)
    assert len(sample) == 2
    assert isinstance(sample, ndarray)
    assert sample.shape == (2, 4)
    sample = space.get_sample(2, True)
    assert len(sample) == 2
    for idx in [0, 1]:
        assert "x1" not in sample[idx]
        assert "x2" not in sample[idx]
        assert isinstance(sample[idx]["y1"], ndarray)
        assert isinstance(sample[idx]["y2"], ndarray)
        assert len(sample[idx]["y1"]) == 1
        assert len(sample[idx]["y2"]) == 3


def test_get_cdf():
    space = ParameterSpace()
    space.add_variable("x1")
    space.add_variable("x2")
    space.add_random_variable("y1", "SPNormalDistribution", mu=0.0, sigma=1.0)
    space.add_random_variable("y2", "SPNormalDistribution", mu=0.0, sigma=1.0, size=3)
    cdf = space.get_cdf({"y1": array([0.0]), "y2": array([0.0] * 3)})
    inv_cdf = space.get_cdf({"y1": array([0.5]), "y2": array([0.5] * 3)}, True)
    assert isinstance(cdf, dict)
    assert isinstance(inv_cdf, dict)
    assert allclose(cdf["y1"], array([0.5]), 1e-3)
    assert allclose(cdf["y2"], array([0.5] * 3), 1e-3)
    assert allclose(inv_cdf["y1"], array([0.0]), 1e-3)
    assert allclose(inv_cdf["y2"], array([0.0] * 3), 1e-3)


def test_range():
    space = ParameterSpace()
    space.add_variable("x1")
    space.add_variable("x2")
    space.add_random_variable("y1", "SPUniformDistribution", lower=0.0, upper=2.0)
    space.add_random_variable("y2", "SPNormalDistribution", mu=0.0, sigma=2.0, size=3)
    expectation = array([0.0, 2.0])
    assert allclose(expectation, space.get_range("y1")[0], 1e-3)
    assert allclose(expectation, space.get_support("y1")[0], 1e-3)
    rng = space.get_range("y2")
    assert len(rng) == 3
    assert allclose(rng[0][1], -rng[0][0])
    assert allclose(rng[1][1], -rng[1][0])
    assert allclose(rng[2][1], -rng[2][0])


def test_normalize():
    space = ParameterSpace()
    space.add_variable("x1")
    space.add_variable("x2")
    space.add_random_variable("y1", "SPUniformDistribution", lower=0.0, upper=2.0)
    space.add_random_variable("y2", "SPNormalDistribution", mu=0.0, sigma=2.0, size=3)
    vector = array([0.5] * 6)
    u_vector = space.normalize_vect(vector)
    expectation = array([0.5] * 2 + [0.25] + [0.598706] * 3)
    assert allclose(u_vector, expectation, 1e-3)


def test_unnormalize():
    space = ParameterSpace()
    space.add_variable("x1")
    space.add_variable("x2")
    space.add_random_variable("y1", "SPUniformDistribution", lower=0.0, upper=2.0)
    space.add_random_variable("y2", "SPNormalDistribution", mu=0.0, sigma=2.0, size=3)
    u_vector = array([0.5] * 2 + [0.25] + [0.598706] * 3)
    vector = space.unnormalize_vect(u_vector)
    expectation = array([0.5] * 6)
    assert allclose(vector, expectation, 1e-3)


def test_update_parameter_space():
    space = ParameterSpace()
    space.add_variable("x1", l_b=0.0, u_b=1.0)
    assert space.get_lower_bound("x1")[0] == 0.0
    assert space.get_upper_bound("x1")[0] == 1.0
    space.add_random_variable("x1", "OTUniformDistribution", 1, lower=0.0, upper=2.0)
    assert space.get_lower_bound("x1")[0] == 0.0
    assert space.get_upper_bound("x1")[0] == 2.0


def test_set_dependence():
    with pytest.raises(NotImplementedError):
        ParameterSpace().set_dependence(["x_1", "x_2"], "gaussian")


def test_str():
    space = ParameterSpace(copula=ComposedDistribution.INDEPENDENT_COPULA)
    space.add_variable("x")
    space.add_random_variable("y", "SPNormalDistribution", mu=0.0, sigma=1.0)
    space.add_random_variable("z", "SPUniformDistribution", lower=0.0, upper=1.0)
    assert "Parameter space" in str(space)


def test_unnormalize_vect():
    space = ParameterSpace()
    space.add_random_variable(
        "x", "SPTriangularDistribution", lower=0.0, mode=0.5, upper=2.0
    )
    assert allclose(space.unnormalize_vect(array([0.5])), array([2.0 - 1.5 ** 0.5]))
    assert space.unnormalize_vect(array([0.5]), use_dist=False)[0] == 1.0


def test_normalize_vect():
    space = ParameterSpace()
    space.add_random_variable(
        "x", "SPTriangularDistribution", lower=0.0, mode=0.5, upper=2.0
    )
    assert allclose(space.normalize_vect(array([2.0 - 1.5 ** 0.5])), array([0.5]))
    assert space.normalize_vect(array([1.0]), use_dist=True)[0] == 0.5


def test_get_cdf_raising_errors():
    space = ParameterSpace()
    space.add_random_variable(
        "x", "SPTriangularDistribution", lower=0.0, mode=0.5, upper=2.0
    )
    value = {"x": 1}
    with pytest.raises(TypeError):
        space.get_cdf(value, inverse=True)
    value = {"x": array([0.5] * 2)}
    with pytest.raises(ValueError):
        space.get_cdf(value, inverse=True)
    value = {"x": array([1.5])}
    with pytest.raises(ValueError):
        space.get_cdf(value, inverse=True)
