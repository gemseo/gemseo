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
from gemseo.core.dataset import Dataset
from gemseo.mlearning.qual_measure.mse_measure import MSEMeasure
from gemseo.mlearning.qual_measure.r2_measure import R2Measure
from gemseo.mlearning.qual_measure.rmse_measure import RMSEMeasure
from gemseo.mlearning.regression.linreg import LinearRegressor
from gemseo.mlearning.regression.polyreg import PolynomialRegressor
from gemseo.problems.dataset.rosenbrock import RosenbrockDataset
from gemseo.utils.testing import compare_dict_of_arrays
from numpy import array
from numpy import linspace
from numpy import newaxis


@pytest.mark.parametrize(
    "method",
    ["bootstrap", "kfolds"],
)
def test_resampling_based_measure(method):
    """Check that a resampling-based measure does not re-train the algo (but a copy)."""
    dataset = RosenbrockDataset(opt_naming=False)
    algo = PolynomialRegressor(dataset, degree=2)
    measure = MSEMeasure(algo)
    measure.evaluate(method)
    assert list(algo.learning_samples_indices) == list(range(len(dataset)))


@pytest.fixture(scope="module")
def learning_dataset() -> Dataset:
    """A learning dataset with 20 points equispaced along the different features."""
    data = linspace(0.0, 1.0, 20)[:, newaxis]
    dataset = Dataset()
    for name in ["x1", "x2"]:
        dataset.add_variable(name, data, group="inputs")
    for name in ["y1", "y2"]:
        dataset.add_variable(name, data, group="outputs")
    return dataset


@pytest.fixture(scope="module")
def test_dataset() -> Dataset:
    """A test dataset with 5 points equispaced along the different features."""
    data = linspace(0.0, 1.0, 5)[:, newaxis]
    dataset = Dataset()
    for name in ["x1", "x2"]:
        dataset.add_variable(name, data, group="inputs")
    for name in ["y1", "y2"]:
        dataset.add_variable(name, data, group="outputs")
    return dataset


@pytest.mark.parametrize("input_names", [None, ["x1"]])
@pytest.mark.parametrize("output_names", [None, ["y2"]])
@pytest.mark.parametrize("method", ["bootstrap", "kfolds", "test"])
@pytest.mark.parametrize(
    "measure_cls,expected", [(MSEMeasure, 0.0), (RMSEMeasure, 0.0), (R2Measure, 1.0)]
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
    if method == "test":
        kwargs["test_data"] = test_dataset

    algo = LinearRegressor(
        learning_dataset, input_names=input_names, output_names=output_names
    )
    if not (measure_cls == R2Measure and method == "bootstrap"):
        measure = measure_cls(algo)
        result = measure.evaluate(method=method, **kwargs)
        assert result == pytest.approx(expected)
        result = measure.evaluate(
            method=method, multioutput=True, as_dict=True, **kwargs
        )
        assert compare_dict_of_arrays(
            result,
            {k: array([expected]) for k in output_names or ["y1", "y2"]},
            tolerance=1e-3,
        )
        result = measure.evaluate(
            method=method, multioutput=False, as_dict=True, **kwargs
        )
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
