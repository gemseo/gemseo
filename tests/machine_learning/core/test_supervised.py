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
#                         documentation
#        :author: Syver Doving Agdestein
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Test supervised machine learning model module."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest
from numpy import arange
from numpy import array
from numpy import array_equal
from numpy import ndarray
from numpy.ma.testutils import assert_close
from numpy.testing import assert_equal

from gemseo.algos.design_space import DesignSpace
from gemseo.datasets.io_dataset import IODataset
from gemseo.machine_learning.core.models.supervised import BaseMLSupervisedModel
from gemseo.machine_learning.core.models.supervised_settings import (
    BaseMLSupervisedModelSettings,
)
from gemseo.machine_learning.regression.models.linreg import LinearRegressor
from gemseo.machine_learning.regression.models.linreg_settings import (
    LinearRegressor_Settings,
)
from gemseo.machine_learning.transformers.dimension_reduction.base_dimension_reduction import (  # noqa: E501
    BaseDimensionReduction,
)
from gemseo.machine_learning.transformers.dimension_reduction.pca import PCA
from gemseo.utils.testing.helpers import assert_exception
from gemseo.utils.testing.helpers import concretize_classes

if TYPE_CHECKING:
    from gemseo.datasets.dataset import Dataset


@pytest.fixture
def io_dataset() -> IODataset:
    """The dataset used to train the supervised machine learning models."""
    data = arange(60).reshape(10, 6)
    variables = ["x_1", "x_2", "y_1"]
    variable_name_to_n_components = {"x_1": 1, "x_2": 2, "y_1": 3}
    variable_name_to_group_name = {"x_1": "inputs", "x_2": "inputs", "y_1": "outputs"}
    dataset = IODataset.from_array(
        data, variables, variable_name_to_n_components, variable_name_to_group_name
    )
    dataset.name = "dataset_name"
    return dataset


def test_constructor(io_dataset) -> None:
    """Test construction."""
    with concretize_classes(BaseMLSupervisedModel):
        ml_model = BaseMLSupervisedModel(io_dataset)

    assert ml_model.algo is None
    assert ml_model.input_names == io_dataset.get_variable_names("inputs")
    assert ml_model.output_names == io_dataset.get_variable_names("outputs")
    design_space = DesignSpace()
    design_space.add_variable("x_1", lower_bound=0.0, upper_bound=54.0)
    design_space.add_variable(
        "x_2", size=2, lower_bound=array([1.0, 2.0]), upper_bound=array([55.0, 56.0])
    )
    assert ml_model.validity_domain == design_space


@pytest.mark.parametrize(
    ("in_transformer", "n_in"), [({}, 3), ({"inputs": PCA(n_components=2)}, 2)]
)
@pytest.mark.parametrize(
    ("out_transformer", "n_out"), [({}, 3), ({"outputs": PCA(n_components=1)}, 1)]
)
def test_get_raw_shapes(
    io_dataset, in_transformer, n_in, out_transformer, n_out
) -> None:
    """Verify the raw input and output shapes of the model."""
    transformer = {}
    transformer.update(in_transformer)
    transformer.update(out_transformer)
    with concretize_classes(BaseMLSupervisedModel):
        model = BaseMLSupervisedModel(
            io_dataset, BaseMLSupervisedModelSettings(transformer=transformer)
        )

    assert model._reduced_input_dimension == n_in
    assert model._reduced_output_dimension == n_out


def test_learn(io_dataset) -> None:
    """Test learn."""
    model = LinearRegressor(io_dataset)
    model.learn()
    reference = model.get_coefficients(False)

    model = LinearRegressor(io_dataset, LinearRegressor_Settings(input_names=["x_1"]))
    model.learn()
    assert not array_equal(model.get_coefficients(False), reference)

    model = LinearRegressor(
        io_dataset, LinearRegressor_Settings(input_names=["x_1", "x_2"])
    )
    model.learn()
    assert array_equal(model.get_coefficients(False), reference)

    model = LinearRegressor(io_dataset, LinearRegressor_Settings(output_names=["y_1"]))
    model.learn()
    assert array_equal(model.get_coefficients(False), reference)

    model = LinearRegressor(io_dataset)
    model.learn(samples=[1, 2])
    assert not array_equal(model.get_coefficients(False), reference)

    model = LinearRegressor(io_dataset)
    model.learn(samples=list(range(10)))
    assert array_equal(model.get_coefficients(False), reference)


def test_io_shape(io_dataset) -> None:
    """Test input output shapes."""
    model = LinearRegressor(io_dataset)
    assert model.input_dimension == 3
    assert model.output_dimension == 3


DICT_1D = {"x_1": array([1.0]), "x_2": array([2.0, 3.0])}
DICT_2D = {"x_1": array([[1.0]]), "x_2": array([[2.0, 3.0]])}
DICT_2D_MULTISAMPLES = {
    "x_1": array([[1.0], [-1.0]]),
    "x_2": array([[2.0, 3.0], [-2.0, -3.0]]),
}
INPUT_VALUE_1D = array([1.0, 2.0, 3.0])
INPUT_VALUE_2D = array([[1.0, 2.0, 3.0]])
INPUT_VALUES = array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])


def test_format_dict(io_dataset) -> None:
    """Test format dict decorator."""
    with concretize_classes(BaseMLSupervisedModel):
        ml_model = BaseMLSupervisedModel(io_dataset)

    partially_transformed = [None]

    @BaseMLSupervisedModel.DataFormatters.format_dict
    def predict_dict(self, input_data):
        """Predict after dict formatting."""
        assert self == ml_model
        partially_transformed[0] = input_data
        return input_data

    out_dict_1d = predict_dict(ml_model, DICT_1D)
    assert array_equal(partially_transformed[0], INPUT_VALUE_1D)
    out_dict_2d = predict_dict(ml_model, DICT_2D)
    assert array_equal(partially_transformed[0], INPUT_VALUE_2D)
    out_dict_2d_multisamples = predict_dict(ml_model, DICT_2D_MULTISAMPLES)
    assert array_equal(partially_transformed[0], INPUT_VALUES)

    out_value_1d = predict_dict(ml_model, INPUT_VALUE_1D)
    assert array_equal(partially_transformed[0], INPUT_VALUE_1D)
    assert array_equal(partially_transformed[0], out_value_1d)
    out_value_2d = predict_dict(ml_model, INPUT_VALUE_2D)
    assert array_equal(partially_transformed[0], INPUT_VALUE_2D)
    assert array_equal(partially_transformed[0], out_value_2d)
    out_values = predict_dict(ml_model, INPUT_VALUES)
    assert array_equal(partially_transformed[0], INPUT_VALUES)
    assert array_equal(partially_transformed[0], out_values)

    assert isinstance(out_dict_1d, dict)
    assert isinstance(out_dict_2d, dict)
    assert isinstance(out_dict_2d_multisamples, dict)
    assert isinstance(out_value_1d, ndarray)
    assert isinstance(out_value_2d, ndarray)
    assert isinstance(out_values, ndarray)

    assert array_equal(out_dict_1d["y_1"], out_value_1d)
    assert array_equal(out_dict_2d["y_1"], out_value_2d)
    assert array_equal(out_dict_2d_multisamples["y_1"], out_values)


def test_format_sample(io_dataset) -> None:
    """Test format sample decorator."""
    partially_transformed = [None]
    with concretize_classes(BaseMLSupervisedModel):
        ml_model = BaseMLSupervisedModel(io_dataset)

    @BaseMLSupervisedModel.DataFormatters.format_samples()
    def predict_sample(self, input_data):
        """Predict (identity function)."""
        assert self == ml_model
        partially_transformed[0] = input_data
        return input_data

    out_value_1d = predict_sample(ml_model, INPUT_VALUE_1D)
    assert array_equal(partially_transformed[0], INPUT_VALUE_1D[None])

    out_value_2d = predict_sample(ml_model, INPUT_VALUE_2D)
    assert array_equal(partially_transformed[0], INPUT_VALUE_2D)

    out_values = predict_sample(ml_model, INPUT_VALUES)
    assert array_equal(partially_transformed[0], INPUT_VALUES)

    assert array_equal(out_value_1d, INPUT_VALUE_1D)
    assert array_equal(out_value_2d, INPUT_VALUE_2D)
    assert array_equal(out_values, INPUT_VALUES)


@pytest.fixture(scope="module")
def dataset_for_transform() -> IODataset:
    """A dataset to check that DataFormatter format_transform()."""
    data = IODataset()
    data.add_variable("x1", array([[0.0], [2.0]]), data.INPUT_GROUP)
    data.add_variable("x2", array([[0.0], [2.0]]), data.INPUT_GROUP)
    data.add_variable("y1", array([[0.0], [4.0]]), data.OUTPUT_GROUP)
    data.add_variable("y2", array([[0.0], [4.0]]), data.OUTPUT_GROUP)
    return data


class NewSupervisedModel(BaseMLSupervisedModel):
    """A supervised model without fitting modelrithm."""

    settings_class = BaseMLSupervisedModelSettings

    def _fit(self, input_data, output_data) -> None:
        return

    def _predict(self, input_data):
        return input_data


@pytest.mark.parametrize(
    (
        "transform_inputs",
        "transform_outputs",
        "transform_in_key",
        "transform_out_key",
        "expected",
    ),
    [
        (False, False, "inputs", "outputs", array([[0.0, 0.0], [2.0, 2.0]])),
        (False, False, "x1", "outputs", array([[0.0, 0.0], [2.0, 2.0]])),
        (False, False, "inputs", "y1", array([[0.0, 0.0], [2.0, 2.0]])),
        (False, False, "x1", "y1", array([[0.0, 0.0], [2.0, 2.0]])),
        (False, True, "inputs", "outputs", array([[0.0, 0.0], [8.0, 8.0]])),
        (False, True, "x1", "outputs", array([[0.0, 0.0], [8.0, 8.0]])),
        (False, True, "inputs", "y1", array([[0.0, 0.0], [8.0, 2.0]])),
        (False, True, "x1", "y1", array([[0.0, 0.0], [8.0, 2.0]])),
        (True, False, "inputs", "outputs", array([[0.0, 0.0], [1.0, 1.0]])),
        (True, False, "x1", "outputs", array([[0.0, 0.0], [1.0, 2.0]])),
        (True, False, "inputs", "y1", array([[0.0, 0.0], [1.0, 1.0]])),
        (True, False, "x1", "y1", array([[0.0, 0.0], [1.0, 2.0]])),
        (True, True, "inputs", "outputs", array([[0.0, 0.0], [4.0, 4.0]])),
        (True, True, "x1", "outputs", array([[0.0, 0.0], [4.0, 8.0]])),
        (True, True, "inputs", "y1", array([[0.0, 0.0], [4.0, 1.0]])),
        (True, True, "x1", "y1", array([[0.0, 0.0], [4.0, 2.0]])),
    ],
)
def test_format_transform(
    dataset_for_transform,
    transform_inputs,
    transform_outputs,
    transform_in_key,
    transform_out_key,
    expected,
) -> None:
    """Check the DataFormatter format_transform().

    This formatter replaces a function by a composition of functions:
    1. transforms the input data,
    2. evaluate the original function,
    3. untransforms the output data.

    Args:
        dataset_for_transform: The dataset used by the ML model.
        transform_inputs: Whether to transform the input data
            before calling the original function.
        transform_outputs: Whether to untransform the output data
            after calling the original function.
        transform_in_key: The name of the input to transform or the input group.
        transform_out_key: The name of the output to transform or the output group.
        expected: The untransformed output data.
    """
    # 1. Define the transformer: MinMaxScaler for an {in,out}put name or group.
    transformer = {transform_in_key: "MinMaxScaler", transform_out_key: "MinMaxScaler"}

    # 2. Train a supervised model.
    model = NewSupervisedModel(
        dataset_for_transform, BaseMLSupervisedModelSettings(transformer=transformer)
    )
    model.learn()

    # 3. Create the DataFormatter to format the prediction method of that model.
    format_function = model.DataFormatters.format_transform(
        transform_inputs, transform_outputs
    )
    # 4. For ease of understanding, we consider the identity as prediction method.
    # its input and output data are supposed to be formatted data
    # if I/O formatters are available.
    predict = lambda self, x: x  # noqa: E731
    formatted_identity_function = format_function(predict)

    # 5. Check the value
    input_data = dataset_for_transform.get_view(group_names="inputs").to_numpy()
    assert_equal(formatted_identity_function(model, input_data), expected)


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    """A learning dataset for the function y=x."""
    data = IODataset()
    data.add_variable("x", array([[1.0], [2.0]]), data.INPUT_GROUP)
    data.add_variable("y", array([[1.0], [2.0]]), data.OUTPUT_GROUP)
    return data


@pytest.mark.parametrize(
    "name", [IODataset.INPUT_GROUP, IODataset.OUTPUT_GROUP, "x", "y"]
)
@pytest.mark.parametrize("fit_transformers", [False, True])
def test_fit_transformers_option(dataset, name, fit_transformers) -> None:
    """Check that the fit_transformers option is correctly used."""
    with concretize_classes(BaseMLSupervisedModel):
        model = BaseMLSupervisedModel(
            dataset, BaseMLSupervisedModelSettings(transformer={name: "MinMaxScaler"})
        )

    model._fit = lambda x, y: None
    model.learn(fit_transformers=fit_transformers)
    assert all(model.transformer[name].offset == -1) is fit_transformers


@pytest.mark.parametrize(
    ("name", "expected"), [("x", {"x": 3, "y": 1}), ("y", {"x": 1, "y": 3})]
)
def test_compute_transformed_variable_sizes(dataset, name, expected) -> None:
    """Check that the compute_transformed_variable_sizes method works."""
    with concretize_classes(BaseMLSupervisedModel, BaseDimensionReduction):
        model = BaseMLSupervisedModel(
            dataset,
            BaseMLSupervisedModelSettings(
                transformer={name: BaseDimensionReduction(n_components=3)}
            ),
        )

    model._BaseMLSupervisedModel__compute_transformed_variable_sizes()
    sizes = model._transformed_variable_sizes
    assert sizes == expected
    assert model._transformed_input_sizes == {"x": sizes["x"]}
    assert model._transformed_output_sizes == {"y": sizes["y"]}


def test_crossed_transformer_failure(dataset, snapshot) -> None:
    """Check that a crossed transformer cannot be applied to outputs."""
    with concretize_classes(BaseMLSupervisedModel):
        model = BaseMLSupervisedModel(
            dataset, BaseMLSupervisedModelSettings(transformer={"y": "PLS"})
        )

    re.escape(
        "The transformer PLS cannot be applied to the outputs "
        "to build a supervised machine learning model."
    )
    with assert_exception(NotImplementedError, snapshot):
        model.learn()


def test_crossed_transformer(dataset) -> None:
    """Check that a crossed transformer can be applied to inputs."""
    with concretize_classes(BaseMLSupervisedModel):
        model = BaseMLSupervisedModel(
            dataset, BaseMLSupervisedModelSettings(transformer={"x": "PLS"})
        )

    model._fit = lambda x, y: None
    model.learn()
    assert_close(model.transformer["x"].algo.x_weights_, array([[1.0]]))

    with concretize_classes(BaseMLSupervisedModel):
        model = BaseMLSupervisedModel(
            dataset, BaseMLSupervisedModelSettings(transformer=model.transformer)
        )

    model._fit = lambda x, y: None
    model.learn(fit_transformers=False)
    assert_close(model.transformer["x"].algo.x_weights_, array([[1.0]]))
