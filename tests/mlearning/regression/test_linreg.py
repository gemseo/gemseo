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

import re
from typing import TYPE_CHECKING

import pytest
from numpy import allclose
from numpy import array
from numpy.testing import assert_almost_equal
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from gemseo.algos.design_space import DesignSpace
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.mlearning.regression.algos.linreg import LinearRegressor
from gemseo.mlearning.transformers.dimension_reduction.pca import PCA
from gemseo.mlearning.transformers.dimension_reduction.pls import PLS
from gemseo.mlearning.transformers.scaler.min_max_scaler import MinMaxScaler
from gemseo.scenarios.doe_scenario import DOEScenario

if TYPE_CHECKING:
    from gemseo.datasets.io_dataset import IODataset

LEARNING_SIZE = 9


@pytest.fixture
def dataset() -> IODataset:
    """The dataset used to train the regression algorithms."""
    discipline = AnalyticDiscipline({"y_1": "1+2*x_1+3*x_2", "y_2": "-1-2*x_1-3*x_2"})
    discipline.set_cache(discipline.CacheType.MEMORY_FULL)
    design_space = DesignSpace()
    design_space.add_variable("x_1", lower_bound=0.0, upper_bound=1.0)
    design_space.add_variable("x_2", lower_bound=0.0, upper_bound=1.0)
    scenario = DOEScenario(
        [discipline], "y_1", design_space, formulation_name="DisciplinaryOpt"
    )
    scenario.execute(algo_name="PYDOE_FULLFACT", n_samples=LEARNING_SIZE)
    return discipline.cache.to_dataset("dataset_name")


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


def test_constructor(dataset) -> None:
    """Test construction."""
    model_ = LinearRegressor(dataset)
    assert model_.algo is not None
    assert model_.SHORT_ALGO_NAME == "LinReg"
    assert model_.LIBRARY == "scikit-learn"


@pytest.mark.parametrize(
    ("l2_penalty_ratio", "type_"),
    [(0.0, Lasso), (1.0, Ridge), (0.5, ElasticNet)],
)
def test_constructor_penalty(dataset, l2_penalty_ratio, type_) -> None:
    """Test construction."""
    model_ = LinearRegressor(
        dataset, penalty_level=0.1, l2_penalty_ratio=l2_penalty_ratio
    )
    assert isinstance(model_.algo, type_)
    model_.learn()
    assert model_._predict(array([[1, 2]])).shape == (1, 2)


def test_learn(dataset) -> None:
    """Test learn."""
    model_ = LinearRegressor(dataset)
    model_.learn()
    assert model_.algo is not None


def test_coefficients(model) -> None:
    """Test coefficients."""
    assert model.coefficients.shape[0] == 2
    assert model.coefficients.shape[1] == 2
    coefficients = model.get_coefficients()
    assert allclose(coefficients["y_1"][0]["x_1"], array([2.0]))
    assert allclose(coefficients["y_1"][0]["x_2"], array([3.0]))
    assert allclose(coefficients["y_2"][0]["x_1"], array([-2.0]))
    assert allclose(coefficients["y_2"][0]["x_2"], array([-3.0]))


@pytest.mark.parametrize(
    (
        "key",
        "transformer",
        "input_dimension",
        "output_dimension",
        "reduced_input_dimension",
        "reduced_output_dimension",
    ),
    [
        ("outputs", PCA(n_components=1), 2, 2, 2, 1),
        ("y_1", PCA(n_components=1), 2, 2, 2, 2),
        ("inputs", PCA(n_components=1), 2, 2, 1, 2),
        ("x_1", PCA(n_components=1), 2, 2, 2, 2),
        ("outputs", MinMaxScaler(), 2, 2, 2, 2),
        ("y_1", MinMaxScaler(), 2, 2, 2, 2),
        ("inputs", MinMaxScaler(), 2, 2, 2, 2),
        ("x_1", MinMaxScaler(), 2, 2, 2, 2),
    ],
)
def test_reduced_io_dimensions(
    dataset,
    key,
    transformer,
    input_dimension,
    output_dimension,
    reduced_input_dimension,
    reduced_output_dimension,
):
    """Check the reduced input and output dimensions."""
    regressor = LinearRegressor(dataset, transformer={key: transformer})
    assert regressor.input_dimension == input_dimension
    assert regressor.output_dimension == output_dimension
    assert regressor._reduced_input_dimension == reduced_input_dimension
    assert regressor._reduced_output_dimension == reduced_output_dimension


def test_coefficients_with_transform(dataset, model_with_transform) -> None:
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
        match=re.escape(
            "Coefficients are only representable in dictionary "
            "form if the transformers do not change the "
            "dimensions of the variables."
        ),
    ):
        model_with_pca.get_coefficients()


def test_intercept(model) -> None:
    """Check the value returned by intercept when as_dict is True."""
    intercept = model.get_intercept()
    assert intercept["y_1"] == pytest.approx([1.0])
    assert intercept["y_2"] == pytest.approx([-1.0])


def test_intercept_false(model) -> None:
    """Check the value returned by intercept when as_dict is False."""
    assert_almost_equal(model.get_intercept(False), array([1.0, -1.0]))


def test_intercept_with_output_dimension_change(dataset) -> None:
    """Verify that an error is raised."""
    model = LinearRegressor(dataset, transformer={"outputs": PCA(n_components=2)})
    model.learn()
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Intercept is only representable in dictionary "
            "form if the transformers do not change the "
            "dimensions of the output variables."
        ),
    ):
        model.get_intercept()


def test_prediction(model) -> None:
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


def test_prediction_with_transform(model_with_transform) -> None:
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


def test_prediction_with_pls(dataset) -> None:
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


def test_prediction_with_pls_failure(dataset) -> None:
    """Test that PLS does not work with output group."""
    model = LinearRegressor(dataset, transformer={"outputs": PLS(n_components=2)})
    with pytest.raises(
        NotImplementedError,
        match=re.escape(
            "The transformer PLS cannot be applied to the outputs "
            "to build a supervised machine learning algorithm."
        ),
    ):
        model.learn()


def test_prediction_jacobian(model) -> None:
    """Test jacobian prediction."""
    input_value = {"x_1": array([1.0]), "x_2": array([2.0])}
    jac = model.predict_jacobian(input_value)
    assert isinstance(jac, dict)
    assert allclose(jac["y_1"]["x_1"], array([[2.0]]))
    assert allclose(jac["y_1"]["x_2"], array([[3.0]]))
    assert allclose(jac["y_2"]["x_1"], array([[-2.0]]))
    assert allclose(jac["y_2"]["x_2"], array([[-3.0]]))


def test_jacobian_transform(model_with_transform) -> None:
    """Test jacobian prediction."""
    input_value = {"x_1": array([1.0]), "x_2": array([2.0])}
    jac = model_with_transform.predict_jacobian(input_value)
    assert isinstance(jac, dict)
    assert allclose(jac["y_1"]["x_1"], array([[2.0]]))
    assert allclose(jac["y_1"]["x_2"], array([[3.0]]))
    assert allclose(jac["y_2"]["x_1"], array([[-2.0]]))
    assert allclose(jac["y_2"]["x_2"], array([[-3.0]]))
