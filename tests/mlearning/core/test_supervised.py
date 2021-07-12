# -*- coding: utf-8 -*-
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
"""Test supervised machine learning algorithm module."""
from __future__ import division, unicode_literals

import pytest
from numpy import arange, array, array_equal, ndarray, zeros

from gemseo.core.dataset import Dataset
from gemseo.mlearning.core.supervised import MLSupervisedAlgo
from gemseo.mlearning.regression.linreg import LinearRegression
from gemseo.mlearning.transform.dimension_reduction.pca import PCA
from gemseo.mlearning.transform.scaler.scaler import Scaler


@pytest.fixture
def io_dataset():  # type: (...) -> Dataset
    """The dataset used to train the supervised machine learning algorithms."""
    data = arange(60).reshape(10, 6)
    variables = ["x_1", "x_2", "y_1"]
    sizes = {"x_1": 1, "x_2": 2, "y_1": 3}
    groups = {"x_1": "inputs", "x_2": "inputs", "y_1": "outputs"}
    dataset = Dataset("dataset_name")
    dataset.set_from_array(data, variables, sizes, groups)
    return dataset


def test_constructor(io_dataset):
    """Test construction."""
    ml_algo = MLSupervisedAlgo(io_dataset)
    assert ml_algo.algo is None
    assert ml_algo.input_names == io_dataset.get_names("inputs")
    assert ml_algo.output_names == io_dataset.get_names("outputs")


@pytest.mark.parametrize(
    "in_transformer,n_in", [({}, 3), ({"inputs": PCA(n_components=2)}, 2)]
)
@pytest.mark.parametrize(
    "out_transformer,n_out", [({}, 3), ({"outputs": PCA(n_components=1)}, 1)]
)
def test_get_raw_shapes(io_dataset, in_transformer, n_in, out_transformer, n_out):
    """Verify the raw input and output shapes of the algorithm."""
    transformer = {}
    transformer.update(in_transformer)
    transformer.update(out_transformer)
    algo = MLSupervisedAlgo(io_dataset, transformer=transformer)
    assert algo._get_raw_shapes() == (n_in, n_out)


def test_notimplementederror(io_dataset):
    """Test that learn() and predict() raise NotImplementedErrors."""
    ml_algo = MLSupervisedAlgo(io_dataset)
    with pytest.raises(NotImplementedError):
        ml_algo.learn()
    with pytest.raises(NotImplementedError):
        ml_algo.predict({"x_1": zeros(1), "x_2": zeros(2)})


def test_learn(io_dataset):
    """Test learn."""
    model = LinearRegression(io_dataset)
    model.learn()
    reference = model.get_coefficients(False)

    model = LinearRegression(io_dataset, input_names=["x_1"])
    model.learn()
    assert not array_equal(model.get_coefficients(False), reference)

    model = LinearRegression(io_dataset, input_names=["x_1", "x_2"])
    model.learn()
    assert array_equal(model.get_coefficients(False), reference)

    model = LinearRegression(io_dataset, output_names=["y_1"])
    model.learn()
    assert array_equal(model.get_coefficients(False), reference)

    model = LinearRegression(io_dataset)
    model.learn(samples=[1, 2])
    assert not array_equal(model.get_coefficients(False), reference)

    model = LinearRegression(io_dataset)
    model.learn(samples=list(range(10)))
    assert array_equal(model.get_coefficients(False), reference)


def test_io_shape(io_dataset):
    """Test input output shapes."""
    model = LinearRegression(io_dataset)
    assert model.input_shape == 3
    assert model.output_shape == 3


DICT_1D = {"x_1": array([1.0]), "x_2": array([2.0, 3.0])}
DICT_2D = {"x_1": array([[1.0]]), "x_2": array([[2.0, 3.0]])}
DICT_2D_MULTISAMPLES = {
    "x_1": array([[1.0], [-1.0]]),
    "x_2": array([[2.0, 3.0], [-2.0, -3.0]]),
}
INPUT_VALUE_1D = array([1.0, 2.0, 3.0])
INPUT_VALUE_2D = array([[1.0, 2.0, 3.0]])
INPUT_VALUES = array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])


def test_format_dict(io_dataset):
    """Test format dict decorator."""
    ml_algo = MLSupervisedAlgo(io_dataset)
    partially_transformed = [None]

    @MLSupervisedAlgo.DataFormatters.format_dict
    def predict_dict(self, input_data):
        """Predict after dict formatting."""
        assert self == ml_algo
        partially_transformed[0] = input_data
        return input_data

    out_dict_1d = predict_dict(ml_algo, DICT_1D)
    assert array_equal(partially_transformed[0], INPUT_VALUE_1D)
    out_dict_2d = predict_dict(ml_algo, DICT_2D)
    assert array_equal(partially_transformed[0], INPUT_VALUE_2D)
    out_dict_2d_multisamples = predict_dict(ml_algo, DICT_2D_MULTISAMPLES)
    assert array_equal(partially_transformed[0], INPUT_VALUES)

    out_value_1d = predict_dict(ml_algo, INPUT_VALUE_1D)
    assert array_equal(partially_transformed[0], INPUT_VALUE_1D)
    assert array_equal(partially_transformed[0], out_value_1d)
    out_value_2d = predict_dict(ml_algo, INPUT_VALUE_2D)
    assert array_equal(partially_transformed[0], INPUT_VALUE_2D)
    assert array_equal(partially_transformed[0], out_value_2d)
    out_values = predict_dict(ml_algo, INPUT_VALUES)
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


def test_format_sample(io_dataset):
    """Test format sample decorator."""
    partially_transformed = [None]
    ml_algo = MLSupervisedAlgo(io_dataset)

    @MLSupervisedAlgo.DataFormatters.format_samples
    def predict_sample(self, input_data):
        """Predict (identity function)."""
        assert self == ml_algo
        partially_transformed[0] = input_data
        return input_data

    out_value_1d = predict_sample(ml_algo, INPUT_VALUE_1D)
    assert array_equal(partially_transformed[0], INPUT_VALUE_1D[None])

    out_value_2d = predict_sample(ml_algo, INPUT_VALUE_2D)
    assert array_equal(partially_transformed[0], INPUT_VALUE_2D)

    out_values = predict_sample(ml_algo, INPUT_VALUES)
    assert array_equal(partially_transformed[0], INPUT_VALUES)

    assert array_equal(out_value_1d, INPUT_VALUE_1D)
    assert array_equal(out_value_2d, INPUT_VALUE_2D)
    assert array_equal(out_values, INPUT_VALUES)


def test_format_transform(io_dataset):
    """Test format transform decorators."""

    class LearnableMLSupervisedAlgo(MLSupervisedAlgo):
        """Supervised algorithm that can learn."""

        def _fit(self, input_data, output_data):
            """Fit data."""
            assert input_data.shape == (10, 3)
            assert output_data.shape == (10, 3)

        def _predict(self, input_data):
            """Predict."""
            return input_data

    partially_transformed = [None]
    transformer = {
        Dataset.INPUT_GROUP: Scaler(offset=5),
        Dataset.OUTPUT_GROUP: Scaler(offset=3),
    }
    ml_algo = LearnableMLSupervisedAlgo(io_dataset, transformer=transformer)
    ml_algo.learn()

    @MLSupervisedAlgo.DataFormatters.format_transform(
        transform_inputs=False, transform_outputs=False
    )
    def predict_transform_none(self, input_data):
        assert self == ml_algo
        partially_transformed[0] = input_data
        return input_data

    @MLSupervisedAlgo.DataFormatters.format_transform(transform_inputs=False)
    def predict_transform_outputs(self, input_data):
        assert self == ml_algo
        partially_transformed[0] = input_data
        return input_data

    @MLSupervisedAlgo.DataFormatters.format_transform(transform_outputs=False)
    def predict_transform_inputs(self, input_data):
        assert self == ml_algo
        partially_transformed[0] = input_data
        return input_data

    @MLSupervisedAlgo.DataFormatters.format_transform()
    def predict_transform_both(self, input_data):
        assert self == ml_algo
        partially_transformed[0] = input_data
        return input_data

    for input_data in [INPUT_VALUE_1D, INPUT_VALUE_2D, INPUT_VALUES]:
        output_data = predict_transform_none(ml_algo, input_data)
        assert array_equal(input_data, partially_transformed[0])
        assert array_equal(input_data, output_data)

        output_data = predict_transform_inputs(ml_algo, input_data)
        assert array_equal(input_data + 5, partially_transformed[0])
        assert array_equal(input_data + 5, output_data)

        output_data = predict_transform_outputs(ml_algo, input_data)
        assert array_equal(input_data, partially_transformed[0])
        assert array_equal(input_data - 3, output_data)

        output_data = predict_transform_both(ml_algo, input_data)
        assert array_equal(input_data + 5, partially_transformed[0])
        assert array_equal(input_data + 5 - 3, output_data)
