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
from numpy import zeros
from scipy.interpolate import RBFInterpolator

from gemseo.dataset.io_dataset import IODataset
from gemseo.discipline.analytic import AnalyticDiscipline
from gemseo.doe.pydoe.settings.pydoe_fullfact import PYDOE_FULLFACT_Settings
from gemseo.machine_learning.regression.model.rbf import RBFRegressor
from gemseo.machine_learning.regression.model.rbf_settings import RBF
from gemseo.machine_learning.regression.model.rbf_settings import RBFRegressor_Settings
from gemseo.scenario.mdo import MDOScenario
from gemseo.space.design import DesignSpace
from gemseo.util.testing.helper import assert_exception

if TYPE_CHECKING:
    from gemseo.dataset.dataset import Dataset

LEARNING_SIZE = 9

INPUT_VALUE = {"x_1": array([1.0]), "x_2": array([2.0])}
INPUT_VALUES = {
    "x_1": array([[0.0], [0.0], [1.0], [2.0]]),
    "x_2": array([[0.0], [1.0], [2.0], [2.0]]),
}


@pytest.fixture
def dataset() -> Dataset:
    """The dataset used to train the regression models."""
    discipline = AnalyticDiscipline({
        "y_1": "1+2*x_1+3*x_2",
        "y_2": "-1-2*x_1-3*x_2",
        "y_3": "3",
    })
    discipline.set_cache(discipline.CacheType.MEMORY_FULL)
    design_space = DesignSpace()
    design_space.add_variable("x_1", lower_bound=0.0, upper_bound=1.0)
    design_space.add_variable("x_2", lower_bound=0.0, upper_bound=1.0)
    scenario = MDOScenario([discipline], design_space)
    scenario.add_objective("y_1")
    scenario.execute(PYDOE_FULLFACT_Settings(n_samples=LEARNING_SIZE))
    return discipline.cache.to_dataset("dataset_name")


@pytest.fixture
def model(dataset) -> RBFRegressor:
    """A trained RBFRegressor."""
    rbf = RBFRegressor(dataset)
    rbf.learn()
    return rbf


@pytest.fixture
def model_with_1d_output(dataset) -> RBFRegressor:
    """A trained RBFRegressor with y_1 as output."""
    rbf = RBFRegressor(dataset, RBFRegressor_Settings(output_names=["y_1"]))
    rbf.learn()
    return rbf


@pytest.mark.parametrize("kernel", RBF)
def test_available_kernels(kernel) -> None:
    """Test that RBFInterpolator accepts the RBF kernel."""
    x = array([[0.0], [0.5], [1.0]])
    RBFInterpolator(x, zeros(3), kernel=kernel, epsilon=1.0)


def test_constructor(dataset) -> None:
    """Test construction."""
    model_ = RBFRegressor(dataset)
    assert model_.algo is None
    assert model_.SHORT_NAME == "RBF"
    assert model_.LIBRARY == "SciPy"


def test_jacobian_not_implemented(dataset, snapshot) -> None:
    """Test that the Jacobian is not implemented for a local interpolant."""
    rbf = RBFRegressor(dataset, RBFRegressor_Settings(neighbors=3))
    rbf.learn()
    with assert_exception(NotImplementedError, snapshot):
        rbf.predict_jacobian(INPUT_VALUE)


def test_learn(dataset) -> None:
    """Test learn."""
    model_ = RBFRegressor(dataset)
    model_.learn()
    assert model_.algo is not None


@pytest.mark.parametrize(
    ("kernel", "expected_epsilon"),
    [(RBF.GAUSSIAN, 3.0), (RBF.CUBIC, 1.0)],
)
def test_epsilon_default(dataset, kernel, expected_epsilon) -> None:
    """Test the default value of the shape parameter epsilon.

    For a scale-dependent kernel,
    it is the reciprocal of the legacy Rbf default,
    i.e. `1/(prod(extents)/n_samples)**(1/dimension)`;
    here `1/(1*1/9)**(1/2) = 3`.
    For a scale-invariant kernel, it is 1 (SciPy default).
    """
    model_ = RBFRegressor(dataset, RBFRegressor_Settings(kernel=kernel))
    model_.learn()
    assert model_.algo.epsilon == pytest.approx(expected_epsilon)


def test_epsilon_default_degenerate() -> None:
    """Test the default value of epsilon when all input dimensions are degenerate."""
    dataset_ = IODataset()
    dataset_.add_input_group(array([[1.0, 2.0]]), ["x_1", "x_2"])
    dataset_.add_output_group(array([[3.0]]), ["y"])
    model_ = RBFRegressor(
        dataset_, RBFRegressor_Settings(kernel=RBF.GAUSSIAN, degree=-1)
    )
    model_.learn()
    assert model_.algo.epsilon == 1.0


def test_smoothing(dataset) -> None:
    """Test that the smoothing setting is passed to RBFInterpolator."""
    model_ = RBFRegressor(dataset, RBFRegressor_Settings(smoothing=0.1))
    model_.learn()
    assert (model_.algo.smoothing == 0.1).all()


def test_degree(dataset) -> None:
    """Test that degree=-1 removes the polynomial term."""
    model_ = RBFRegressor(dataset, RBFRegressor_Settings(degree=-1))
    model_.learn()
    assert model_.algo.powers.size == 0
    assert model_.predict(INPUT_VALUE)["y_1"].shape == (1,)
    assert model_.predict_jacobian(INPUT_VALUE)["y_1"]["x_1"].shape == (1, 1)


def test_neighbors(dataset) -> None:
    """Test the prediction of a local interpolant."""
    model_ = RBFRegressor(dataset, RBFRegressor_Settings(neighbors=5))
    model_.learn()
    prediction = model_.predict(INPUT_VALUE)
    assert allclose(prediction["y_1"], -prediction["y_2"])


def test_kernel(model) -> None:
    """Test the kernel property."""
    assert model.kernel == RBF.MULTIQUADRIC


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


@pytest.mark.parametrize("kernel", RBF)
def test_predict_jacobian(dataset, kernel) -> None:
    """Test prediction."""
    model_ = RBFRegressor(dataset, RBFRegressor_Settings(kernel=kernel))
    model_.learn()
    jacobian = model_.predict_jacobian(INPUT_VALUE)
    jacobians = model_.predict_jacobian(INPUT_VALUES)
    assert isinstance(jacobian, dict)
    assert isinstance(jacobians, dict)
    assert allclose(jacobian["y_1"]["x_1"], -jacobian["y_2"]["x_1"])
    assert allclose(jacobian["y_1"]["x_2"], -jacobian["y_2"]["x_2"])
    assert allclose(jacobians["y_1"]["x_1"], -jacobians["y_2"]["x_1"])
    assert allclose(jacobians["y_1"]["x_2"], -jacobians["y_2"]["x_2"])
