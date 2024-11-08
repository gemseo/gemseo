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
from numpy import hstack
from numpy import ndarray
from numpy.testing import assert_almost_equal
from numpy.testing import assert_equal

from gemseo.algos.design_space import DesignSpace
from gemseo.datasets.io_dataset import IODataset
from gemseo.disciplines.analytic import AnalyticDiscipline
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
    discipline.set_cache(discipline.CacheType.MEMORY_FULL)
    design_space = DesignSpace()
    design_space.add_variable("x_1", lower_bound=0.0, upper_bound=1.0)
    design_space.add_variable("x_2", lower_bound=0.0, upper_bound=1.0)
    scenario = DOEScenario(
        [discipline], "y_1", design_space, formulation_name="DisciplinaryOpt"
    )
    scenario.execute(algo_name="PYDOE_FULLFACT", n_samples=LEARNING_SIZE)
    return discipline.cache.to_dataset("dataset_name")


@pytest.fixture(params=[{}, GaussianProcessRegressor.DEFAULT_TRANSFORMER])
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


@pytest.mark.parametrize(
    ("bounds", "expected"),
    [
        ((), [(0.01, 100.0), (0.01, 100.0)]),
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
    ("compute_options", "init_options", "expected_samples", "expected_samples_1pt"),
    [
        (
            {},
            {},
            array([
                [
                    [1.85016387, -2.00702945],
                    [4.72740681, -4.50629058],
                    [4.00618543, -3.87720141],
                ],
                [
                    [1.95208716, -1.97554591],
                    [4.57660313, -4.63186214],
                    [3.8260905, -4.06281621],
                ],
            ]),
            array([[1.90124246, -2.07677922], [1.9260292, -1.81711579]]),
        ),
        (
            {"seed": 2},
            {},
            array([
                [
                    [1.94456792, -1.92438423],
                    [4.63585051, -4.52883111],
                    [3.86614502, -3.96826468],
                ],
                [
                    [1.88691454, -1.88091603],
                    [4.77084545, -4.72290722],
                    [4.00805521, -3.91364315],
                ],
            ]),
            array([[1.90124246, -2.07677922], [1.9260292, -1.81711579]]),
        ),
        (
            {},
            {"output_names": ["y_1"]},
            array([
                [[1.85016387], [4.72740681], [4.00618543]],
                [[1.95208716], [4.57660313], [3.8260905]],
            ]),
            array([[1.90124246], [1.92602921]]),
        ),
    ],
)
@pytest.mark.parametrize("transformer", [{}, {"inputs": "Scaler", "outputs": "Scaler"}])
def test_compute_samples(
    dataset,
    compute_options,
    init_options,
    expected_samples,
    expected_samples_1pt,
    transformer,
):
    """Test the method compute_samples."""
    model = GaussianProcessRegressor(dataset, transformer=transformer, **init_options)
    model.learn()
    input_data = array([[0.23, 0.19], [0.73, 0.69], [0.13, 0.89]])
    samples = model.compute_samples(input_data, 2, **compute_options)
    assert_almost_equal(samples, expected_samples)

    samples = model.compute_samples(input_data[0], 2, **compute_options)
    assert_almost_equal(samples, expected_samples_1pt)


def test_std_multiple_output():
    """Check the standard deviation when the number of outputs is greater than 1."""
    x = array([[0.0], [0.5], [1.0]])
    y = hstack((x**2, 10 * x**2))
    dataset = IODataset()
    dataset.add_input_group(x, variable_names="x")
    dataset.add_output_group(y, variable_names="y")
    gpr = GaussianProcessRegressor(dataset)
    gpr.learn()
    std = gpr.predict_std(array([0.25]))[0]
    assert std[0] != std[1]
    assert std[0] == pytest.approx(std[1] / 10, rel=1e-9)


def test_homonymous_io():
    """Check that a supervised ML algo can use with homonymous inputs and outputs."""
    x = array([[0.0], [0.5], [1.0]])
    y = x**2
    dataset = IODataset()
    dataset.add_input_group(x, variable_names="x")
    dataset.add_output_group(y, variable_names="y")
    gpr = GaussianProcessRegressor(dataset)
    gpr.learn()
    reference = gpr.predict(array([0.25]))

    dataset = IODataset()
    dataset.add_input_group(x)
    dataset.add_output_group(y)
    gpr = GaussianProcessRegressor(dataset)
    gpr.learn()
    assert_equal(gpr.predict(array([0.25])), reference)
