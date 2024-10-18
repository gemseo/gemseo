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
from __future__ import annotations

import re

import pytest
from numpy import array
from numpy.testing import assert_equal

from gemseo.mlearning.regression.algos.linreg import LinearRegressor
from gemseo.mlearning.regression.algos.polyreg import PolynomialRegressor
from gemseo.mlearning.regression.algos.regressor_chain import RegressorChain
from gemseo.mlearning.regression.quality.mse_measure import MSEMeasure


@pytest.fixture(scope="module")
def algo(dataset) -> RegressorChain:
    """A regressor chain model for the Rosenbrock dataset."""
    algo = RegressorChain(dataset)
    algo.add_algo("LinearRegressor")
    algo.add_algo("PolynomialRegressor", degree=2)
    return algo


@pytest.fixture(scope="module")
def standard_algo(algo) -> RegressorChain:
    """A polynomial chain model with degree 2."""
    algo.learn()
    return algo


@pytest.fixture(scope="module")
def improved_algo(algo) -> RegressorChain:
    """A polynomial chain model with degree 4."""
    algo.add_algo("PolynomialRegressor", degree=4)
    algo.learn()
    return algo


def test_no_regressors(dataset):
    """Check the ValueError message raised when the chain contains no regressor."""
    algo = RegressorChain(dataset)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The regressor chain contains no regressor; "
            "please add regressors using the add_algo method."
        ),
    ):
        algo.learn()


def test_standard_algo(standard_algo):
    """Check that the regressor chain model is not accurate."""
    assert MSEMeasure(standard_algo).compute_learning_measure() != pytest.approx(0.0)


def test_improved_algo(improved_algo):
    """Check that an improved regressor chain model is not accurate."""
    assert MSEMeasure(improved_algo).compute_learning_measure() == pytest.approx(0.0)


def test_sub_algos(improved_algo):
    """Check the properties of the sub-algorithms."""
    algo_0, algo_1, algo_2 = improved_algo._RegressorChain__algos
    assert isinstance(algo_0, LinearRegressor)
    assert isinstance(algo_1, PolynomialRegressor)
    assert isinstance(algo_2, PolynomialRegressor)
    assert algo_1._settings.degree == 2
    assert algo_2._settings.degree == 4


def test_output_data(improved_algo):
    """Check that the model returns the sum of the output data of the sub-algos."""
    algo_0, algo_1, algo_2 = improved_algo._RegressorChain__algos
    data = array([1.0, 1.0])
    output_data = algo_0.predict(data) + algo_1.predict(data) + algo_2.predict(data)
    assert output_data == improved_algo.predict(data)


def test_jacobian_data(improved_algo):
    """Check that the model returns the sum of the jacobian data of the sub-algos."""
    algo_0, algo_1, algo_2 = improved_algo._RegressorChain__algos
    data = array([1.0, 1.0])
    output_data = (
        algo_0.predict_jacobian(data)
        + algo_1.predict_jacobian(data)
        + algo_2.predict_jacobian(data)
    )
    assert_equal(output_data, improved_algo.predict_jacobian(data))
