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

import pytest
from numpy import array
from numpy.testing import assert_equal

from gemseo.machine_learning.regression.models.linreg import LinearRegressor
from gemseo.machine_learning.regression.models.linreg_settings import (
    LinearRegressor_Settings,
)
from gemseo.machine_learning.regression.models.polyreg import PolynomialRegressor
from gemseo.machine_learning.regression.models.polyreg_settings import (
    PolynomialRegressor_Settings,
)
from gemseo.machine_learning.regression.models.regressor_chain import RegressorChain
from gemseo.machine_learning.regression.quality.mse_measure import MSEMeasure
from gemseo.utils.testing.helpers import assert_exception


@pytest.fixture(scope="module")
def model(dataset) -> RegressorChain:
    """A regressor chain model for the Rosenbrock dataset."""
    model = RegressorChain(dataset)
    model.add_regressor(LinearRegressor_Settings())
    model.add_regressor(PolynomialRegressor_Settings(degree=2))
    return model


@pytest.fixture(scope="module")
def standard_model(model) -> RegressorChain:
    """A polynomial chain model with degree 2."""
    model.learn()
    return model


@pytest.fixture(scope="module")
def improved_model(model) -> RegressorChain:
    """A polynomial chain model with degree 4."""
    model.add_regressor(PolynomialRegressor_Settings(degree=4))
    model.learn()
    return model


def test_no_regressors(dataset, snapshot):
    """Check the ValueError message raised when the chain contains no regressor."""
    model = RegressorChain(dataset)
    with assert_exception(ValueError, snapshot):
        model.learn()


def test_standard_model(standard_model):
    """Check that the regressor chain model is not accurate."""
    assert MSEMeasure(standard_model).compute_learning_measure() != pytest.approx(0.0)


def test_improved_model(improved_model):
    """Check that an improved regressor chain model is not accurate."""
    assert MSEMeasure(improved_model).compute_learning_measure() == pytest.approx(0.0)


def test_sub_models(improved_model):
    """Check the properties of the sub-models."""
    model_0, model_1, model_2 = improved_model._RegressorChain__regressors
    assert isinstance(model_0, LinearRegressor)
    assert isinstance(model_1, PolynomialRegressor)
    assert isinstance(model_2, PolynomialRegressor)
    assert model_1._settings.degree == 2
    assert model_2._settings.degree == 4


def test_output_data(improved_model):
    """Check that the model returns the sum of the output data of the sub-models."""
    model_0, model_1, model_2 = improved_model._RegressorChain__regressors
    data = array([1.0, 1.0])
    output_data = model_0.predict(data) + model_1.predict(data) + model_2.predict(data)
    assert output_data == improved_model.predict(data)


def test_jacobian_data(improved_model):
    """Check that the model returns the sum of the jacobian data of the sub-models."""
    model_0, model_1, model_2 = improved_model._RegressorChain__regressors
    data = array([1.0, 1.0])
    output_data = (
        model_0.predict_jacobian(data)
        + model_1.predict_jacobian(data)
        + model_2.predict_jacobian(data)
    )
    assert_equal(output_data, improved_model.predict_jacobian(data))
