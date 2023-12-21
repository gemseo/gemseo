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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import re

import pytest
from numpy import array
from numpy import concatenate
from numpy.testing import assert_allclose

from gemseo.core.parallel_execution.disc_parallel_execution import DiscParallelExecution
from gemseo.datasets.io_dataset import IODataset
from gemseo.disciplines.surrogate import SurrogateDiscipline
from gemseo.mlearning.quality_measures.r2_measure import R2Measure
from gemseo.mlearning.regression.linreg import LinearRegressor
from gemseo.post.mlearning.ml_regressor_quality_viewer import MLRegressorQualityViewer
from gemseo.utils.comparisons import compare_dict_of_arrays
from gemseo.utils.repr_html import REPR_HTML_WRAPPER


@pytest.fixture(scope="module")
def dataset():
    """The input-output dataset to train the surrogate discipline."""
    x_1 = array([[0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1]]).T
    x_2 = array([[0, 0.5, 1, 0, 0.5, 1, 0, 0.5, 1]]).T
    y_1 = 1 + 2 * x_1 + 3 * x_2
    y_2 = -1 - 2 * x_1 - 3 * x_2
    dataset = IODataset(dataset_name="func")
    dataset.add_input_variable("x_1", x_1)
    dataset.add_input_variable("x_2", x_2)
    dataset.add_output_variable("y_1", y_1)
    dataset.add_output_variable("y_2", y_2)
    return dataset


@pytest.fixture(scope="module")
def linear_discipline(dataset) -> SurrogateDiscipline:
    """A surrogate discipline relying on a linear regressor."""
    return SurrogateDiscipline("LinearRegressor", dataset)


def test_instantiation_without_data(dataset):
    """Check that instantiation without existing model and data raises an error."""
    with pytest.raises(
        ValueError, match=re.escape("data is required to train the surrogate model.")
    ):
        SurrogateDiscipline("LinearRegressor")


def test_linearization_mode_with_gradient(linear_discipline):
    """Check the attribute linearization_mode for a model with gradient."""
    assert linear_discipline.linearization_mode == "auto"


def test_linearization_mode_without_gradient(dataset):
    """Check the attribute linearization_mode for a model without gradient."""
    discipline = SurrogateDiscipline("GaussianProcessRegressor", dataset)
    assert discipline.linearization_mode == "finite_differences"
    assert {"x_1", "x_2"} == set(discipline.get_input_data_names())
    assert {"y_1", "y_2"} == set(discipline.get_output_data_names())


def test_instantiation_from_algo(dataset):
    """Check the instantiation from an MLRegressionAlgo."""
    algo = LinearRegressor(dataset)
    algo.learn()
    discipline = SurrogateDiscipline(algo)
    assert discipline.linearization_mode == "auto"
    assert {"x_1", "x_2"} == set(discipline.get_input_data_names())
    assert {"y_1", "y_2"} == set(discipline.get_output_data_names())


def test_repr_str(linear_discipline):
    """Check the __repr__ and __str__ of a surrogate discipline."""
    expected = """Surrogate discipline: LinReg_func
   Dataset name: func
   Dataset size: 9
   Surrogate model: LinearRegressor
   Inputs: x_1, x_2
   Outputs: y_1, y_2
   Linearization mode: auto"""
    assert repr(linear_discipline) == expected
    assert str(linear_discipline) == "LinReg_func"


def test_execute(linear_discipline):
    """Check the execution of a surrogate discipline."""
    linear_discipline.execute()
    assert compare_dict_of_arrays(
        linear_discipline.get_output_data(),
        {"y_1": array([3.5]), "y_2": array([-3.5])},
        tolerance=1e-6,
    )


def test_linearize(linear_discipline):
    """Check the computation of the Jacobian of a surrogate discipline."""
    assert compare_dict_of_arrays(
        linear_discipline.linearize(),
        {
            "y_1": {"x_1": array([[2.0]]), "x_2": array([[3.0]])},
            "y_2": {"x_1": array([[-2.0]]), "x_2": array([[-3.0]])},
        },
        tolerance=1e-6,
    )


def test_parallel_execute(linear_discipline, dataset):
    """Test the execution of the surrogate discipline in parallel."""
    other_linear_discipline = SurrogateDiscipline("LinearRegressor", dataset)

    parallel_execution = DiscParallelExecution(
        [linear_discipline, other_linear_discipline], n_processes=2
    )
    parallel_execution.execute([
        {"x_1": array([0.5]), "x_2": array([0.5])},
        {"x_1": array([1.0]), "x_2": array([1.0])},
    ])

    assert_allclose(
        concatenate(list(linear_discipline.get_outputs_by_name(["y_1", "y_2"]))),
        array([3.5, -3.5]),
        atol=1e-3,
    )
    assert_allclose(
        concatenate(list(other_linear_discipline.get_outputs_by_name(["y_1", "y_2"]))),
        array([6.0, -6.0]),
        atol=1e-3,
    )


def test_serialize(linear_discipline, tmp_wd):
    """Check the serialization of a surrogate discipline."""
    file_path = "discipline.pkl"
    linear_discipline.to_pickle(file_path)

    loaded_discipline = SurrogateDiscipline.from_pickle(file_path)
    loaded_discipline.execute()

    assert linear_discipline.local_data == loaded_discipline.local_data


def test_get_error_measure(linear_discipline):
    """Check that get_error_measure returns an instance of MLErrorMeasure."""
    error_measure = linear_discipline.get_error_measure("R2Measure")
    assert isinstance(error_measure, R2Measure)
    assert error_measure.algo == linear_discipline.regression_model


def test_repr_html(dataset):
    """Check SurrogateDiscipline._repr_html_."""
    assert SurrogateDiscipline(
        "LinearRegressor", dataset
    )._repr_html_() == REPR_HTML_WRAPPER.format(
        "Surrogate discipline: LinReg_func<br/>"
        "<ul>"
        "<li>Dataset name: func</li>"
        "<li>Dataset size: 9</li>"
        "<li>Surrogate model: LinearRegressor</li>"
        "<li>Inputs: x_1, x_2</li>"
        "<li>Outputs: y_1, y_2</li>"
        "<li>Linearization mode: auto</li>"
        "</ul>"
    )


def test_get_quality_viewer(dataset):
    """Check the method get_quality_viewer()."""
    discipline = SurrogateDiscipline("LinearRegressor", dataset)
    quality_viewer = discipline.get_quality_viewer()
    assert isinstance(quality_viewer, MLRegressorQualityViewer)
    assert quality_viewer._MLRegressorQualityViewer__algo == discipline.regression_model
