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
"""Test linear regression module."""
from __future__ import annotations

import pytest
from gemseo.algos.design_space import DesignSpace
from gemseo.core.dataset import Dataset
from gemseo.core.doe_scenario import DOEScenario
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.mlearning.api import import_regression_model
from gemseo.mlearning.regression.linreg import LinearRegressor
from gemseo.mlearning.transform.dimension_reduction.pca import PCA
from gemseo.mlearning.transform.dimension_reduction.pls import PLS
from gemseo.mlearning.transform.scaler.min_max_scaler import MinMaxScaler
from numpy import allclose
from numpy import array
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

LEARNING_SIZE = 9


@pytest.fixture
def dataset() -> Dataset:
    """The dataset used to train the regression algorithms."""
    discipline = AnalyticDiscipline({"y_1": "1+2*x_1+3*x_2", "y_2": "-1-2*x_1-3*x_2"})
    discipline.set_cache_policy(discipline.MEMORY_FULL_CACHE)
    design_space = DesignSpace()
    design_space.add_variable("x_1", l_b=0.0, u_b=1.0)
    design_space.add_variable("x_2", l_b=0.0, u_b=1.0)
    scenario = DOEScenario([discipline], "DisciplinaryOpt", "y_1", design_space)
    scenario.execute({"algo": "fullfact", "n_samples": LEARNING_SIZE})
    return discipline.cache.export_to_dataset("dataset_name")


@pytest.fixture
def model(dataset) -> LinearRegressor:
    """A trained LinearRegressor."""
    linreg = LinearRegressor(dataset)
    linreg.learn()
    return linreg


@pytest.fixture
def model_with_transform(dataset) -> LinearRegressor:
    """A trained LinearRegressor with inputs and outputs scaling."""
    linreg = LinearRegressor(
        dataset, transformer={"inputs": MinMaxScaler(), "outputs": MinMaxScaler()}
    )
    linreg.learn()
    return linreg


def test_constructor(dataset):
    """Test construction."""
    model_ = LinearRegressor(dataset)
    assert model_.algo is not None
    assert model_.SHORT_ALGO_NAME == "LinReg"
    assert model_.LIBRARY == "scikit-learn"


@pytest.mark.parametrize(
    "l2_penalty_ratio,type",
    [(0.0, Lasso), (1.0, Ridge), (0.5, ElasticNet)],
)
def test_constructor_penalty(dataset, l2_penalty_ratio, type):
    """Test construction."""
    model_ = LinearRegressor(
        dataset, penalty_level=0.1, l2_penalty_ratio=l2_penalty_ratio
    )
    assert isinstance(model_.algo, type)
    model_.learn()
    assert model_._predict(array([[1, 2]])).shape == (1, 2)


def test_learn(dataset):
    """Test learn."""
    model_ = LinearRegressor(dataset)
    model_.learn()
    assert model_.algo is not None


def test_coefficients(model):
    """Test coefficients."""
    assert model.coefficients.shape[0] == 2
    assert model.coefficients.shape[1] == 2
    coefficients = model.get_coefficients()
    assert allclose(coefficients["y_1"][0]["x_1"], array([2.0]))
    assert allclose(coefficients["y_1"][0]["x_2"], array([3.0]))
    assert allclose(coefficients["y_2"][0]["x_1"], array([-2.0]))
    assert allclose(coefficients["y_2"][0]["x_2"], array([-3.0]))


def test_coefficients_with_transform(dataset, model_with_transform):
    """Test correct handling of get_coefficients with transformers."""
    model_with_transform.get_coefficients(as_dict=False)
    model_with_transform.get_coefficients(as_dict=True)

    model_with_pca = LinearRegressor(
        dataset, transformer={dataset.OUTPUT_GROUP: PCA(n_components=1)}
    )
    model_with_pca.learn()
    model_with_pca.get_coefficients(as_dict=False)
    with pytest.raises(
        ValueError,
        match=(
            "Coefficients are only representable in dictionary "
            "form if the transformers do not change the "
            "dimensions of the variables."
        ),
    ):
        model_with_pca.get_coefficients()


def test_intercept(model):
    """Test intercept."""
    intercept = model.get_intercept()
    assert allclose(intercept["y_1"], array([1.0]))
    assert allclose(intercept["y_2"], array([-1.0]))


def test_intercept_with_output_dimension_change(dataset):
    """Verify that an error is raised."""
    model = LinearRegressor(dataset, transformer={"outputs": PCA(n_components=2)})
    model.learn()
    with pytest.raises(
        ValueError,
        match=(
            "Intercept is only representable in dictionary "
            "form if the transformers do not change the "
            "dimensions of the output variables."
        ),
    ):
        model.get_intercept()


def test_prediction(model):
    """Test prediction."""
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
    """Test prediction."""
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


def test_prediction_with_pls(dataset):
    """Test prediction."""
    model = LinearRegressor(dataset, transformer={"inputs": PLS(n_components=2)})
    model.learn()
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


def test_prediction_with_pls_failure(dataset):
    """Test that PLS does not work with output group."""
    model = LinearRegressor(dataset, transformer={"outputs": PLS(n_components=2)})
    with pytest.raises(
        NotImplementedError,
        match=(
            "The transformer PLS cannot be applied to the outputs "
            "to build a supervised machine learning algorithm."
        ),
    ):
        model.learn()


def test_prediction_jacobian(model):
    """Test jacobian prediction."""
    input_value = {"x_1": array([1.0]), "x_2": array([2.0])}
    jac = model.predict_jacobian(input_value)
    assert isinstance(jac, dict)
    assert allclose(jac["y_1"]["x_1"], array([[2.0]]))
    assert allclose(jac["y_1"]["x_2"], array([[3.0]]))
    assert allclose(jac["y_2"]["x_1"], array([[-2.0]]))
    assert allclose(jac["y_2"]["x_2"], array([[-3.0]]))


def test_jacobian_transform(model_with_transform):
    """Test jacobian prediction."""
    input_value = {"x_1": array([1.0]), "x_2": array([2.0])}
    jac = model_with_transform.predict_jacobian(input_value)
    assert isinstance(jac, dict)
    assert allclose(jac["y_1"]["x_1"], array([[2.0]]))
    assert allclose(jac["y_1"]["x_2"], array([[3.0]]))
    assert allclose(jac["y_2"]["x_1"], array([[-2.0]]))
    assert allclose(jac["y_2"]["x_2"], array([[-3.0]]))


def test_save_and_load(model, tmp_wd):
    """Test save and load."""
    dirname = model.save()
    imported_model = import_regression_model(dirname)
    input_value = {"x_1": array([1.0]), "x_2": array([2.0])}
    out1 = model.predict(input_value)
    out2 = imported_model.predict(input_value)
    for name, value in out1.items():
        assert allclose(value, out2[name], 1e-3)
