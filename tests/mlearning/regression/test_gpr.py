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
"""Test Gaussian process regression algorithm module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from numpy import allclose
from numpy import array
from numpy import array_equal
from numpy import ndarray
from numpy.testing import assert_almost_equal

from gemseo.algos.design_space import DesignSpace
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.mlearning import import_regression_model
from gemseo.mlearning.regression.algos.gpr import GaussianProcessRegressor
from gemseo.scenarios.doe_scenario import DOEScenario
from gemseo.utils.data_conversion import concatenate_dict_of_arrays_to_array

if TYPE_CHECKING:
    from gemseo.datasets.dataset import Dataset

LEARNING_SIZE = 9


@pytest.fixture
def dataset() -> Dataset:
    """The dataset used to train the regression algorithms."""
    discipline = AnalyticDiscipline({"y_1": "1+2*x_1+3*x_2", "y_2": "-1-2*x_1-3*x_2"})
    discipline.set_cache_policy(discipline.CacheType.MEMORY_FULL)
    design_space = DesignSpace()
    design_space.add_variable("x_1", l_b=0.0, u_b=1.0)
    design_space.add_variable("x_2", l_b=0.0, u_b=1.0)
    scenario = DOEScenario([discipline], "DisciplinaryOpt", "y_1", design_space)
    scenario.execute({"algo": "fullfact", "n_samples": LEARNING_SIZE})
    return discipline.cache.to_dataset("dataset_name")


@pytest.fixture(params=[None, GaussianProcessRegressor.DEFAULT_TRANSFORMER])
def model(request, dataset) -> GaussianProcessRegressor:
    """A trained GaussianProcessRegressor."""
    gpr = GaussianProcessRegressor(dataset, transformer=request.param)
    gpr.learn()
    return gpr


def test_constructor(dataset) -> None:
    """Test construction."""
    gpr = GaussianProcessRegressor(dataset)
    assert gpr.algo is not None
    assert gpr.SHORT_ALGO_NAME == "GPR"
    assert gpr.LIBRARY == "scikit-learn"


def test_learn(dataset) -> None:
    """Test learn."""
    gpr = GaussianProcessRegressor(dataset)
    gpr.learn()
    assert gpr.algo is not None


def test_predict(model) -> None:
    """Test prediction."""
    input_value = {"x_1": array([1.0]), "x_2": array([2.0])}
    prediction = model.predict(input_value)
    assert isinstance(prediction, dict)
    assert "y_1" in prediction
    assert "y_2" in prediction
    assert isinstance(prediction["y_1"], ndarray)
    assert prediction["y_1"].shape == (1,)
    assert isinstance(prediction["y_2"], ndarray)
    assert prediction["y_2"].shape == (1,)
    assert allclose(prediction["y_1"], -prediction["y_2"], 1e-2)


def test_predict_std_training_point(model) -> None:
    """Test std prediction for a training point."""
    prediction_std = model.predict_std({"x_1": array([1.0]), "x_2": array([1.0])})
    assert allclose(prediction_std, 0, atol=1e-3)
    assert prediction_std.shape == (1, 2)


def test_predict_std_1d_output(dataset) -> None:
    """Test std prediction for a training point with a 1d output."""
    gpr = GaussianProcessRegressor(dataset, output_names=["y_1"])
    gpr.learn()
    prediction_std = gpr.predict_std({"x_1": array([1.0]), "x_2": array([1.0])})
    assert allclose(prediction_std, 0, atol=1e-3)
    assert prediction_std.shape == (1, 1)


def test_predict_std_test_point(model) -> None:
    """Test std prediction for a test point."""
    prediction_std = model.predict_std({"x_1": array([1.0]), "x_2": array([2.0])})
    assert (prediction_std > 0).all()


def test_predict_std_input_array(model) -> None:
    """Test std prediction when the input data is an array."""
    input_value = {"x_1": array([1.0]), "x_2": array([2.0])}
    prediction_std = model.predict_std(input_value)
    input_value = concatenate_dict_of_arrays_to_array(input_value, model.input_names)
    assert array_equal(model.predict_std(input_value), prediction_std)


@pytest.mark.parametrize(
    ("x_1", "x_2"), [([1.0], [2.0]), ([[1.0], [1.0]], [[2.0], [2.0]])]
)
def test_predict_std_shape(model, x_1, x_2) -> None:
    """Test the shape and content of standard deviation."""
    input_value = {"x_1": array(x_1), "x_2": array(x_2)}
    prediction_std = model.predict_std(input_value)
    assert prediction_std.ndim == 2
    assert prediction_std.shape[1] == 2


def test_save_and_load(model, tmp_wd) -> None:
    """Test save and load."""
    dirname = model.to_pickle()
    imported_model = import_regression_model(dirname)
    input_value = {"x_1": array([1.0]), "x_2": array([2.0])}
    out1 = model.predict(input_value)
    out2 = imported_model.predict(input_value)
    for name, value in out1.items():
        assert allclose(value, out2[name], 1e-3)


@pytest.mark.parametrize(
    ("bounds", "expected"),
    [
        (None, [(0.01, 100.0), (0.01, 100.0)]),
        ((0.1, 10), [(0.1, 10), (0.1, 10)]),
        ({"x_2": (0.1, 10)}, [(0.01, 100), (0.1, 10)]),
    ],
)
def test_bounds(dataset, bounds, expected) -> None:
    """Verify that bounds are correctly passed to the default kernel."""
    model = GaussianProcessRegressor(dataset, bounds=bounds)
    assert model.algo.kernel.length_scale_bounds == expected


def test_kernel(dataset) -> None:
    """Verify that the property 'kernel' corresponds to the kernel for prediction."""
    model = GaussianProcessRegressor(dataset)
    assert id(model.kernel) == id(model.algo.kernel)
    model.learn()
    assert id(model.kernel) == id(model.algo.kernel_)


@pytest.mark.parametrize(
    ("compute_options", "init_options", "expected_samples"),
    [
        (
            {},
            {},
            (
                array([
                    [1.793727, 1.6737006],
                    [4.7481523, 4.7274054],
                    [4.1133087, 4.0414895],
                ]),
                array([
                    [-1.9714888, -1.9169749],
                    [-4.610077, -4.6684153],
                    [-3.9723156, -3.9135493],
                ]),
            ),
        ),
        (
            {"seed": 2},
            {},
            (
                array([
                    [1.9019072, 1.8469586],
                    [4.6702036, 4.8555419],
                    [3.88026, 4.0085456],
                ]),
                array([
                    [-1.9195439, -1.826369],
                    [-4.5784073, -4.7704947],
                    [-4.0406025, -3.9173866],
                ]),
            ),
        ),
        (
            {},
            {"output_names": ["y_1"]},
            (
                array([
                    [1.793727, 1.6737006],
                    [4.7481523, 4.7274054],
                    [4.1133087, 4.0414895],
                ]),
            ),
        ),
    ],
)
def test_compute_samples(dataset, compute_options, init_options, expected_samples):
    """Test the method compute_samples."""
    model = GaussianProcessRegressor(dataset, **init_options)
    model.learn()
    input_data = array([[0.23, 0.19], [0.73, 0.69], [0.13, 0.89]])
    samples = model.compute_samples(input_data, 2, **compute_options)
    assert_almost_equal(samples[0], expected_samples[0])
    if not init_options:
        assert_almost_equal(samples[1], expected_samples[1])
