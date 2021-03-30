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
""" Test linear regression module. """

from __future__ import absolute_import, division, unicode_literals

import pytest
from future import standard_library
from numpy import allclose, array
from sklearn.linear_model import ElasticNet, Lasso, Ridge

from gemseo.algos.design_space import DesignSpace
from gemseo.core.analytic_discipline import AnalyticDiscipline
from gemseo.core.doe_scenario import DOEScenario
from gemseo.mlearning.api import import_regression_model
from gemseo.mlearning.regression.linreg import LinearRegression
from gemseo.mlearning.transform.dimension_reduction.pca import PCA
from gemseo.mlearning.transform.scaler.min_max_scaler import MinMaxScaler

standard_library.install_aliases()


LEARNING_SIZE = 9


@pytest.fixture
def dataset():
    """ Dataset from a R^2 -> R^2 function sampled over [0,1]^2. """
    expressions_dict = {"y_1": "1+2*x_1+3*x_2", "y_2": "-1-2*x_1-3*x_2"}
    discipline = AnalyticDiscipline("func", expressions_dict)
    discipline.set_cache_policy(discipline.MEMORY_FULL_CACHE)
    design_space = DesignSpace()
    design_space.add_variable("x_1", l_b=0.0, u_b=1.0)
    design_space.add_variable("x_2", l_b=0.0, u_b=1.0)
    scenario = DOEScenario([discipline], "DisciplinaryOpt", "y_1", design_space)
    scenario.execute({"algo": "fullfact", "n_samples": LEARNING_SIZE})
    return discipline.cache.export_to_dataset("dataset_name")


@pytest.fixture
def model(dataset):
    """ Define model from data. """
    linreg = LinearRegression(dataset)
    linreg.learn()
    return linreg


@pytest.fixture
def model_with_transform(dataset):
    """ Define model with transformers from data. """
    linreg = LinearRegression(
        dataset, transformer={"inputs": MinMaxScaler(), "outputs": MinMaxScaler()}
    )
    linreg.learn()
    return linreg


def test_constructor(dataset):
    """ Test construction."""
    model_ = LinearRegression(dataset)
    assert model_.algo is not None


def test_constructor_penalty(dataset):
    """ Test construction."""
    model_ = LinearRegression(dataset, penalty_level=0.1, l2_penalty_ratio=0.0)
    assert isinstance(model_.algo, Lasso)
    model_ = LinearRegression(dataset, penalty_level=0.1, l2_penalty_ratio=1.0)
    assert isinstance(model_.algo, Ridge)
    model_ = LinearRegression(dataset, penalty_level=0.1, l2_penalty_ratio=0.5)
    assert isinstance(model_.algo, ElasticNet)


def test_learn(dataset):
    """ Test learn."""
    model_ = LinearRegression(dataset)
    model_.learn()
    assert model_.algo is not None


def test_coefficients(model):
    """ Test coefficients. """
    assert model.coefficients.shape[0] == 2
    assert model.coefficients.shape[1] == 2
    coefficients = model.get_coefficients()
    assert allclose(coefficients["y_1"][0]["x_1"], array([2.0]))
    assert allclose(coefficients["y_1"][0]["x_2"], array([3.0]))
    assert allclose(coefficients["y_2"][0]["x_1"], array([-2.0]))
    assert allclose(coefficients["y_2"][0]["x_2"], array([-3.0]))


def test_coefficients_with_transform(dataset, model_with_transform):
    """ Test correct handling of get_coefficients with transformers. """
    model_with_transform.get_coefficients(as_dict=False)
    model_with_transform.get_coefficients(as_dict=True)

    model_with_pca = LinearRegression(
        dataset, transformer={dataset.OUTPUT_GROUP: PCA(n_components=1)}
    )
    model_with_pca.learn()
    model_with_pca.get_coefficients(as_dict=False)
    with pytest.raises(ValueError):
        model_with_pca.get_coefficients(as_dict=True)


def test_intercept(model):
    """ Test intercept. """
    intercept = model.get_intercept()
    assert allclose(intercept["y_1"], array([1.0]))
    assert allclose(intercept["y_2"], array([-1.0]))


def test_prediction(model):
    """ Test prediction. """
    input_value = {"x_1": array([1.0]), "x_2": array([2.0])}
    another_input_value = {
        "x_1": array([[1.0], [0.0], [-1.0]]),
        "x_2": array([[2.0], [0.0], [1.0]]),
    }
    prediction = model.predict(input_value)
    another_prediction = model.predict(another_input_value)
    assert isinstance(prediction, dict)
    assert isinstance(another_prediction, dict)
    assert allclose(prediction["y_1"], array([9.0]))
    assert allclose(prediction["y_2"], array([-9.0]))
    assert allclose(another_prediction["y_1"], array([[9.0], [1.0], [2.0]]))
    assert allclose(another_prediction["y_2"], array([[-9.0], [-1.0], [-2.0]]))


def test_prediction_with_transform(model_with_transform):
    """ Test prediction. """
    input_value = {"x_1": array([1.0]), "x_2": array([2.0])}
    another_input_value = {
        "x_1": array([[1.0], [0.0], [-1.0]]),
        "x_2": array([[2.0], [0.0], [1.0]]),
    }
    prediction = model_with_transform.predict(input_value)
    another_prediction = model_with_transform.predict(another_input_value)
    assert isinstance(prediction, dict)
    assert isinstance(another_prediction, dict)
    assert allclose(prediction["y_1"], array([9.0]))
    assert allclose(prediction["y_2"], array([-9.0]))
    assert allclose(another_prediction["y_1"], array([[9.0], [1.0], [2.0]]))
    assert allclose(another_prediction["y_2"], array([[-9.0], [-1.0], [-2.0]]))


def test_prediction_jacobian(model):
    """ Test jacobian prediction. """
    input_value = {"x_1": array([1.0]), "x_2": array([2.0])}
    jac = model.predict_jacobian(input_value)
    assert isinstance(jac, dict)
    assert allclose(jac["y_1"]["x_1"], array([[2.0]]))
    assert allclose(jac["y_1"]["x_2"], array([[3.0]]))
    assert allclose(jac["y_2"]["x_1"], array([[-2.0]]))
    assert allclose(jac["y_2"]["x_2"], array([[-3.0]]))


def test_jacobian_transform(model_with_transform):
    """ Test jacobian prediction. """
    input_value = {"x_1": array([1.0]), "x_2": array([2.0])}
    jac = model_with_transform.predict_jacobian(input_value)
    assert isinstance(jac, dict)
    assert allclose(jac["y_1"]["x_1"], array([[2.0]]))
    assert allclose(jac["y_1"]["x_2"], array([[3.0]]))
    assert allclose(jac["y_2"]["x_1"], array([[-2.0]]))
    assert allclose(jac["y_2"]["x_2"], array([[-3.0]]))


def test_save_and_load(model, tmp_path):
    """ Test save and load. """
    dirname = model.save(path=str(tmp_path))
    imported_model = import_regression_model(dirname)
    input_value = {"x_1": array([1.0]), "x_2": array([2.0])}
    out1 = model.predict(input_value)
    out2 = imported_model.predict(input_value)
    for name, value in out1.items():
        assert allclose(value, out2[name], 1e-3)
