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

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from numpy import allclose
from numpy import array
from scipy.interpolate.rbf import Rbf

from gemseo.algos.design_space import DesignSpace
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.mlearning.regression.algos.rbf import RBFRegressor
from gemseo.mlearning.regression.algos.rbf_settings import RBF
from gemseo.scenarios.doe_scenario import DOEScenario

if TYPE_CHECKING:
    from gemseo.datasets.dataset import Dataset

LEARNING_SIZE = 9

INPUT_VALUE = {"x_1": array([1.0]), "x_2": array([2.0])}
INPUT_VALUES = {
    "x_1": array([[0.0], [0.0], [1.0], [2.0]]),
    "x_2": array([[0.0], [1.0], [2.0], [2.0]]),
}


@pytest.fixture
def dataset() -> Dataset:
    """The dataset used to train the regression algorithms."""
    discipline = AnalyticDiscipline({
        "y_1": "1+2*x_1+3*x_2",
        "y_2": "-1-2*x_1-3*x_2",
        "y_3": "3",
    })
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
def model(dataset) -> RBFRegressor:
    """A trained RBFRegressor."""
    rbf = RBFRegressor(dataset)
    rbf.learn()
    return rbf


@pytest.fixture
def model_with_custom_function(dataset) -> RBFRegressor:
    """A trained RBFRegressor  f(r) = r**2 - 1 as kernel function."""

    def der_function(input_data, norm_input_data, eps):
        return 2 * input_data / eps**2

    rbf = RBFRegressor(
        dataset, function=(lambda r: r**2 - 1), der_function=der_function
    )
    rbf.learn()
    return rbf


@pytest.fixture
def model_with_1d_output(dataset) -> RBFRegressor:
    """A trained RBFRegressor with y_1 as output."""
    rbf = RBFRegressor(dataset, output_names=["y_1"])
    rbf.learn()
    return rbf


def test_get_available_functions() -> None:
    """Test available RBFs."""
    for function in RBF:
        assert hasattr(Rbf, f"_h_{function}")


def test_constructor(dataset) -> None:
    """Test construction."""
    model_ = RBFRegressor(dataset)
    assert model_.algo is None
    assert model_.SHORT_ALGO_NAME == "RBF"
    assert model_.LIBRARY == "SciPy"


def test_jacobian_not_implemented(dataset) -> None:
    """Test cases where the Jacobian is not implemented."""
    # Test unimplemented norm
    rbf = RBFRegressor(dataset, norm="canberra")
    rbf.learn()
    with pytest.raises(NotImplementedError):
        rbf.predict_jacobian(INPUT_VALUE)

    # Test rbf function without derivative
    rbf = RBFRegressor(dataset, function=(lambda x: x - 5))
    rbf.learn()
    with pytest.raises(NotImplementedError):
        rbf.predict_jacobian(INPUT_VALUE)


def test_learn(dataset) -> None:
    """Test learn."""
    model_ = RBFRegressor(dataset)
    model_.learn()
    assert model_.algo is not None


def test_average(model) -> None:
    """Test average."""
    avg_dict = {"y_1": 3.5, "y_2": -3.5, "y_3": 3}
    y_average = array([0.0, 0.0, 0.0])
    for i in range(3):
        y_average[i] = avg_dict[model.output_names[i]]
    assert allclose(model.y_average, y_average)


def test_prediction(model) -> None:
    """Test prediction."""
    prediction = model.predict(INPUT_VALUE)
    predictions = model.predict(INPUT_VALUES)
    assert isinstance(prediction, dict)
    assert isinstance(predictions, dict)
    assert allclose(prediction["y_1"], -prediction["y_2"])
    assert allclose(predictions["y_1"], -predictions["y_2"])
    assert allclose(prediction["y_3"], 3)
    assert allclose(predictions["y_3"], 3)


def test_prediction_custom(model_with_custom_function) -> None:
    """Test prediction."""
    prediction = model_with_custom_function.predict(INPUT_VALUE)
    predictions = model_with_custom_function.predict(INPUT_VALUES)
    assert isinstance(prediction, dict)
    assert isinstance(predictions, dict)
    assert allclose(prediction["y_1"], -prediction["y_2"])
    assert allclose(predictions["y_1"], -predictions["y_2"])
    assert allclose(prediction["y_3"], 3)
    assert allclose(predictions["y_3"], 3)


def test_pred_single_out(model_with_1d_output) -> None:
    """Test predict with one output variable."""
    prediction = model_with_1d_output.predict(INPUT_VALUE)
    predictions = model_with_1d_output.predict(INPUT_VALUES)
    assert isinstance(prediction, dict)
    assert isinstance(predictions, dict)
    prediction = model_with_1d_output.predict(array([1, 1]))
    predictions = model_with_1d_output.predict(array([[1, 1], [0, 0], [0, 1]]))
    assert prediction.shape == (1,)
    assert predictions.shape == (3, 1)


def test_predict_jacobian(dataset) -> None:
    """Test prediction."""
    for function in RBF:
        model_ = RBFRegressor(dataset, function=function)
        model_.learn()
        jacobian = model_.predict_jacobian(INPUT_VALUE)
        jacobians = model_.predict_jacobian(INPUT_VALUES)
        assert isinstance(jacobian, dict)
        assert isinstance(jacobians, dict)
        assert allclose(jacobian["y_1"]["x_1"], -jacobian["y_2"]["x_1"])
        assert allclose(jacobian["y_1"]["x_2"], -jacobian["y_2"]["x_2"])
        assert allclose(jacobians["y_1"]["x_1"], -jacobians["y_2"]["x_1"])
        assert allclose(jacobians["y_1"]["x_2"], -jacobians["y_2"]["x_2"])


def test_predict_jacobian_custom(model_with_custom_function) -> None:
    """Test prediction."""
    jacobian = model_with_custom_function.predict_jacobian(INPUT_VALUE)
    jacobians = model_with_custom_function.predict_jacobian(INPUT_VALUES)
    assert isinstance(jacobian, dict)
    assert isinstance(jacobians, dict)
    assert allclose(jacobian["y_1"]["x_1"], -jacobian["y_2"]["x_1"])
    assert allclose(jacobian["y_1"]["x_2"], -jacobian["y_2"]["x_2"])
    assert allclose(jacobians["y_1"]["x_1"], -jacobians["y_2"]["x_1"])
    assert allclose(jacobians["y_1"]["x_2"], -jacobians["y_2"]["x_2"])
