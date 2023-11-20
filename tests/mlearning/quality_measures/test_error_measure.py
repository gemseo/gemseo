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
#        :author: Syver Doving Agdestein
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Test error measure module."""

from __future__ import annotations

import pytest
from numpy import array
from numpy import linspace
from numpy import newaxis

from gemseo.datasets.io_dataset import IODataset
from gemseo.mlearning.quality_measures.error_measure_factory import (
    MLErrorMeasureFactory,
)
from gemseo.mlearning.quality_measures.mse_measure import MSEMeasure
from gemseo.mlearning.quality_measures.r2_measure import R2Measure
from gemseo.mlearning.quality_measures.rmse_measure import RMSEMeasure
from gemseo.mlearning.regression.linreg import LinearRegressor
from gemseo.mlearning.regression.polyreg import PolynomialRegressor
from gemseo.problems.dataset.rosenbrock import create_rosenbrock_dataset
from gemseo.utils.comparisons import compare_dict_of_arrays


@pytest.mark.parametrize(
    "method",
    ["compute_bootstrap_measure", "compute_cross_validation_measure"],
)
def test_resampling_based_measure(method):
    """Check that a resampling-based measure does not re-train the algo (but a copy)."""
    dataset = create_rosenbrock_dataset(opt_naming=False)
    algo = PolynomialRegressor(dataset, degree=2)
    measure = MSEMeasure(algo)
    getattr(measure, method)()
    assert list(algo.learning_samples_indices) == list(range(len(dataset)))


@pytest.fixture(scope="module")
def learning_dataset() -> IODataset:
    """A learning dataset with 20 points equispaced along the different features."""
    data = linspace(0.0, 1.0, 20)[:, newaxis]
    dataset = IODataset()
    for name in ["x1", "x2"]:
        dataset.add_variable(name, data, "inputs")
    for name in ["y1", "y2"]:
        dataset.add_variable(name, data, "outputs")
    return dataset


@pytest.fixture()
def linear_regressor(learning_dataset) -> LinearRegressor:
    """A linear regressor."""
    algo = LinearRegressor(learning_dataset)
    algo.learn()
    return algo


@pytest.fixture(scope="module")
def test_dataset() -> IODataset:
    """A test dataset with 5 points equispaced along the different features."""
    data = linspace(0.0, 1.0, 5)[:, newaxis]
    dataset = IODataset()
    for name in ["x1", "x2"]:
        dataset.add_variable(name, data, "inputs")
    for name in ["y1", "y2"]:
        dataset.add_variable(name, data, "outputs")
    return dataset


@pytest.mark.parametrize("input_names", [None, ["x1"]])
@pytest.mark.parametrize("output_names", [None, ["y2"]])
@pytest.mark.parametrize(
    "method",
    [
        "compute_bootstrap_measure",
        "compute_cross_validation_measure",
        "compute_test_measure",
    ],
)
@pytest.mark.parametrize(
    ("measure_cls", "expected"),
    [(MSEMeasure, 0.0), (RMSEMeasure, 0.0), (R2Measure, 1.0)],
)
def test_subset_of_inputs_and_outputs(
    measure_cls,
    expected,
    learning_dataset,
    test_dataset,
    method,
    input_names,
    output_names,
):
    """Check that quality measures correctly handle algo with subsets of IO names."""
    kwargs = {}
    if method == "compute_test_measure":
        kwargs["test_data"] = test_dataset

    algo = LinearRegressor(
        learning_dataset, input_names=input_names, output_names=output_names
    )
    if not (measure_cls == R2Measure and method == "compute_bootstrap_measure"):
        measure = measure_cls(algo)
        evaluate = getattr(measure, method)
        result = evaluate(**kwargs)
        assert result == pytest.approx(expected)
        result = evaluate(multioutput=True, as_dict=True, **kwargs)
        assert compare_dict_of_arrays(
            result,
            {k: array([expected]) for k in output_names or ["y1", "y2"]},
            tolerance=1e-3,
        )
        result = evaluate(multioutput=False, as_dict=True, **kwargs)
        if output_names is not None:
            assert compare_dict_of_arrays(
                result,
                {output_names[0]: array([expected])},
                tolerance=1e-3,
            )
        else:
            assert compare_dict_of_arrays(
                result, {"y1#y2": array([expected])}, tolerance=1e-3
            )


def test_no_resampling_result_storage(linear_regressor):
    """Check that by default, a quality measure does not store the resampling result."""
    mse = MSEMeasure(linear_regressor)
    mse.evaluate_kfolds()
    assert linear_regressor.resampling_results == {}


@pytest.mark.parametrize(
    ("method", "resampler_name", "class_name", "dimension"),
    [
        ("evaluate_kfolds", "CrossValidation", "CrossValidation", 5),
        ("evaluate_loo", "LeaveOneOut", "CrossValidation", 20),
        ("evaluate_bootstrap", "Bootstrap", "Bootstrap", 15),
    ],
)
def test_resampling_result_storage(
    linear_regressor, method, resampler_name, class_name, dimension
):
    """Check that the resampling result can be stored in the ML algorithm and reused."""
    options = {}
    if resampler_name == "Bootstrap":
        options["n_replicates"] = dimension
    mse = MSEMeasure(linear_regressor)
    getattr(mse, method)(store_resampling_result=True, **options)

    # 1. Check the resampling result attached to the ML algorithm.
    resampling_result = linear_regressor.resampling_results[resampler_name]
    assert resampling_result[0].__class__.__name__ == class_name
    algos = resampling_result[1]
    first_algo_id = id(algos[0])
    assert len(algos) == dimension
    for algo in algos:
        assert algo.is_trained

    predictions = resampling_result[2]
    n_samples = len(linear_regressor.learning_set)
    if class_name == "Bootstrap":
        assert len(predictions) == dimension
        for prediction in predictions:
            assert prediction.ndim == 2
            assert prediction.shape[1] == 2
            assert prediction.shape[0] < n_samples
    else:
        assert predictions.shape == (n_samples, 2)

    # 2. Check that a new computation of the measure generates a new resampling result
    # in the case of CrossValidation and Bootstrap
    # because the default seed is random (default value: None).
    # For LeaveOneOut, we use the same seed.
    getattr(mse, method)(store_resampling_result=True, **options)
    algos = linear_regressor.resampling_results[resampler_name][1]
    methods_with_random_seed_by_default = ["CrossValidation", "Bootstrap"]
    if resampler_name in methods_with_random_seed_by_default:
        assert id(algos[0]) != first_algo_id
    else:
        assert id(algos[0]) == first_algo_id

    # 3. Check that a new computation reuses the existing resampling result
    # when using the same seed.
    if resampler_name in methods_with_random_seed_by_default:
        options["seed"] = 1

    getattr(mse, method)(store_resampling_result=True, **options)
    algos = linear_regressor.resampling_results[resampler_name][1]
    first_algo_id = id(algos[0])
    getattr(mse, method)(store_resampling_result=True, **options)
    algos = linear_regressor.resampling_results[resampler_name][1]
    assert id(algos[0]) == first_algo_id


def test_factory():
    """Test the MLErrorMeasureFactory."""
    assert MLErrorMeasureFactory().is_available("R2Measure")
    assert not MLErrorMeasureFactory().is_available("SilhouetteMeasure")
