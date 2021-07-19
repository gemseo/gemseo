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
"""Test machine learning regression algorithm module."""
from __future__ import division, unicode_literals

import pytest
from numpy import allclose, arange, array, zeros

from gemseo.core.dataset import Dataset
from gemseo.mlearning.regression.gpr import GaussianProcessRegression
from gemseo.mlearning.regression.linreg import LinearRegression
from gemseo.mlearning.regression.regression import MLRegressionAlgo


@pytest.fixture
def io_dataset():
    """The dataset used to train the regression algorithms."""
    data = arange(60).reshape(10, 6)
    variables = ["x_1", "x_2", "y_1"]
    sizes = {"x_1": 1, "x_2": 2, "y_1": 3}
    groups = {"x_1": "inputs", "x_2": "inputs", "y_1": "outputs"}
    dataset = Dataset("dataset_name")
    dataset.set_from_array(data, variables, sizes, groups)
    return dataset


def test_notimplementederror(io_dataset):
    """Test not implemented methods."""
    ml_algo = MLRegressionAlgo(io_dataset)
    with pytest.raises(NotImplementedError):
        ml_algo.learn()
    with pytest.raises(NotImplementedError):
        ml_algo.predict({"x_1": zeros(1), "x_2": zeros(2)})
    with pytest.raises(NotImplementedError):
        ml_algo.predict_jacobian({"x_1": zeros(1), "x_2": zeros(2)})


def test_predict(io_dataset):
    """Test prediction."""
    ml_algo = GaussianProcessRegression(io_dataset)
    ml_algo.learn()
    input_data = io_dataset.get_data_by_group("inputs", True)
    input_data = {key: val[0] for key, val in input_data.items()}
    output_data = io_dataset.get_data_by_group("outputs", True)
    output_data = {key: val[0] for key, val in output_data.items()}
    prediction = ml_algo.predict(input_data)
    assert allclose(prediction["y_1"], output_data["y_1"])


def test_predict_jacobian():
    """Test predict Jacobian."""
    data = array(
        [
            [1.0, 2.0, 3.0, 6.0, -6.0],
            [2.0, 3.0, 4.0, 9.0, -9.0],
            [3.0, 4.0, 5.0, 12.0, -12.0],
        ]
    )
    variables = ["x_1", "x_2", "y_1"]
    sizes = {"x_1": 1, "x_2": 2, "y_1": 2}
    groups = {"x_1": "inputs", "x_2": "inputs", "y_1": "outputs"}
    dataset = Dataset("dataset_name")
    dataset.set_from_array(data, variables, sizes, groups)
    ml_algo = LinearRegression(dataset)
    ml_algo.learn()
    jac = ml_algo.predict_jacobian({"x_1": zeros(1), "x_2": zeros(2)})
    assert allclose(jac["y_1"]["x_1"], array([[1.0], [-1.0]]))
    assert allclose(jac["y_1"]["x_2"], array([[1.0, 1.0], [-1.0, -1.0]]))
