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
"""Test radial basis function regression module."""
from __future__ import division, unicode_literals

import pytest
from numpy import allclose, array
from scipy.interpolate.rbf import Rbf

from gemseo.algos.design_space import DesignSpace
from gemseo.core.analytic_discipline import AnalyticDiscipline
from gemseo.core.dataset import Dataset
from gemseo.core.doe_scenario import DOEScenario
from gemseo.mlearning.api import import_regression_model
from gemseo.mlearning.regression.rbf import RBFRegression

LEARNING_SIZE = 9

INPUT_VALUE = {"x_1": array([1.0]), "x_2": array([2.0])}
INPUT_VALUES = {
    "x_1": array([[0.0], [0.0], [1.0], [2.0]]),
    "x_2": array([[0.0], [1.0], [2.0], [2.0]]),
}


@pytest.fixture
def dataset():  # type: (...) -> Dataset
    """The dataset used to train the regression algorithms."""
    expressions_dict = {"y_1": "1+2*x_1+3*x_2", "y_2": "-1-2*x_1-3*x_2", "y_3": "3"}
    discipline = AnalyticDiscipline("func", expressions_dict)
    discipline.set_cache_policy(discipline.MEMORY_FULL_CACHE)
    design_space = DesignSpace()
    design_space.add_variable("x_1", l_b=0.0, u_b=1.0)
    design_space.add_variable("x_2", l_b=0.0, u_b=1.0)
    scenario = DOEScenario([discipline], "DisciplinaryOpt", "y_1", design_space)
    scenario.execute({"algo": "fullfact", "n_samples": LEARNING_SIZE})
    return discipline.cache.export_to_dataset("dataset_name")


@pytest.fixture
def model(dataset):  # type: (...) -> RBFRegression
    """A trained RBFRegression."""
    rbf = RBFRegression(dataset)
    rbf.learn()
    return rbf


@pytest.fixture
def model_with_custom_function(dataset):  # type: (...) -> RBFRegression
    """A trained RBFRegression  f(r) = r**2 - 1 as kernel function."""

    def der_function(input_data, norm_input_data, eps):
        return 2 * input_data / eps ** 2

    rbf = RBFRegression(
        dataset, function=(lambda r: r ** 2 - 1), der_function=der_function
    )
    rbf.learn()
    return rbf


@pytest.fixture
def model_with_1d_output(dataset):  # type: (...) -> RBFRegression
    """A trained RBFRegression with y_1 as output."""
    rbf = RBFRegression(dataset, output_names=["y_1"])
    rbf.learn()
    return rbf


def test_get_available_functions():
    """Test available RBFs."""
    for function in RBFRegression.AVAILABLE_FUNCTIONS:
        assert hasattr(Rbf, "_h_{}".format(function))


def test_constructor(dataset):
    """Test construction."""
    model_ = RBFRegression(dataset)
    assert model_.algo is None


def test_jacobian_not_implemented(dataset):
    """Test cases where the Jacobian is not implemented."""
    # Test unimplemented norm
    rbf = RBFRegression(dataset, norm="canberra")
    rbf.learn()
    with pytest.raises(NotImplementedError):
        rbf.predict_jacobian(INPUT_VALUE)

    # Test rbf function without derivative
    rbf = RBFRegression(dataset, function=(lambda x: x - 5))
    rbf.learn()
    with pytest.raises(NotImplementedError):
        rbf.predict_jacobian(INPUT_VALUE)


def test_learn(dataset):
    """Test learn."""
    model_ = RBFRegression(dataset)
    model_.learn()
    assert model_.algo is not None


def test_average(model):
    """Test average."""
    avg_dict = {"y_1": 3.5, "y_2": -3.5, "y_3": 3}
    y_average = array([0.0, 0.0, 0.0])
    for i in range(3):
        y_average[i] = avg_dict[model.output_names[i]]
    assert allclose(model.y_average, y_average)


def test_prediction(model):
    """Test prediction."""
    prediction = model.predict(INPUT_VALUE)
    predictions = model.predict(INPUT_VALUES)
    assert isinstance(prediction, dict)
    assert isinstance(predictions, dict)
    assert allclose(prediction["y_1"], -prediction["y_2"])
    assert allclose(predictions["y_1"], -predictions["y_2"])
    assert allclose(prediction["y_3"], 3)
    assert allclose(predictions["y_3"], 3)


def test_prediction_custom(model_with_custom_function):
    """Test prediction."""
    prediction = model_with_custom_function.predict(INPUT_VALUE)
    predictions = model_with_custom_function.predict(INPUT_VALUES)
    assert isinstance(prediction, dict)
    assert isinstance(predictions, dict)
    assert allclose(prediction["y_1"], -prediction["y_2"])
    assert allclose(predictions["y_1"], -predictions["y_2"])
    assert allclose(prediction["y_3"], 3)
    assert allclose(predictions["y_3"], 3)


def test_pred_single_out(model_with_1d_output):
    """Test predict with one output variable."""
    prediction = model_with_1d_output.predict(INPUT_VALUE)
    predictions = model_with_1d_output.predict(INPUT_VALUES)
    assert isinstance(prediction, dict)
    assert isinstance(predictions, dict)
    prediction = model_with_1d_output.predict(array([1, 1]))
    predictions = model_with_1d_output.predict(array([[1, 1], [0, 0], [0, 1]]))
    assert prediction.shape == (1,)
    assert predictions.shape == (3, 1)


def test_predict_jacobian(dataset):
    """Test prediction."""
    for function in RBFRegression.AVAILABLE_FUNCTIONS:
        model_ = RBFRegression(dataset, function=function)
        model_.learn()
        jacobian = model_.predict_jacobian(INPUT_VALUE)
        jacobians = model_.predict_jacobian(INPUT_VALUES)
        assert isinstance(jacobian, dict)
        assert isinstance(jacobians, dict)
        assert allclose(jacobian["y_1"]["x_1"], -jacobian["y_2"]["x_1"])
        assert allclose(jacobian["y_1"]["x_2"], -jacobian["y_2"]["x_2"])
        assert allclose(jacobians["y_1"]["x_1"], -jacobians["y_2"]["x_1"])
        assert allclose(jacobians["y_1"]["x_2"], -jacobians["y_2"]["x_2"])


def test_predict_jacobian_custom(model_with_custom_function):
    """Test prediction."""
    jacobian = model_with_custom_function.predict_jacobian(INPUT_VALUE)
    jacobians = model_with_custom_function.predict_jacobian(INPUT_VALUES)
    assert isinstance(jacobian, dict)
    assert isinstance(jacobians, dict)
    assert allclose(jacobian["y_1"]["x_1"], -jacobian["y_2"]["x_1"])
    assert allclose(jacobian["y_1"]["x_2"], -jacobian["y_2"]["x_2"])
    assert allclose(jacobians["y_1"]["x_1"], -jacobians["y_2"]["x_1"])
    assert allclose(jacobians["y_1"]["x_2"], -jacobians["y_2"]["x_2"])


def test_save_and_load(model, tmp_path):
    """Test save and load."""
    dirname = model.save(path=str(tmp_path))
    imported_model = import_regression_model(dirname)
    out1 = model.predict(INPUT_VALUE)
    out2 = imported_model.predict(INPUT_VALUE)
    for name, value in out1.items():
        assert allclose(value, out2[name])
