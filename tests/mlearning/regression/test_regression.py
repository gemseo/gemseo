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
from __future__ import annotations

import re

import pytest
from gemseo.core.dataset import Dataset
from gemseo.mlearning.regression.gpr import GaussianProcessRegressor
from gemseo.mlearning.regression.linreg import LinearRegressor
from numpy import allclose
from numpy import arange
from numpy import array
from numpy import zeros


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


def test_predict(io_dataset):
    """Test prediction."""
    ml_algo = GaussianProcessRegressor(io_dataset)
    ml_algo.learn()
    input_data = io_dataset.get_data_by_group("inputs", True)
    input_data = {key: val[0] for key, val in input_data.items()}
    output_data = io_dataset.get_data_by_group("outputs", True)
    output_data = {key: val[0] for key, val in output_data.items()}
    prediction = ml_algo.predict(input_data)
    assert allclose(prediction["y_1"], output_data["y_1"])


@pytest.fixture(scope="module")
def dataset_for_jacobian() -> Dataset:
    """The dataset used to check the Jacobian computation."""
    samples = array(
        [
            [1.0, 2.0, 3.0, 6.0, -6.0],
            [2.0, 3.0, 4.0, 9.0, -9.0],
            [3.0, 4.0, 5.0, 12.0, -12.0],
        ]
    )
    variables = ["x_1", "x_2", "y_1"]
    sizes = {"x_1": 1, "x_2": 2, "y_1": 2}
    groups = {"x_1": "inputs", "x_2": "inputs", "y_1": "outputs"}
    data = Dataset("dataset_name")
    data.set_from_array(samples, variables, sizes, groups)
    return data


@pytest.mark.parametrize(
    "groups", [None, ["inputs"], ["outputs"], ["inputs", "outputs"]]
)
def test_predict_jacobian(dataset_for_jacobian, groups):
    """Test predict Jacobian."""
    if not groups:
        transformer = None
    else:
        transformer = {group: "MinMaxScaler" for group in groups}
    ml_algo = LinearRegressor(dataset_for_jacobian, transformer=transformer)
    ml_algo.learn()
    jac = ml_algo.predict_jacobian({"x_1": zeros(1), "x_2": zeros(2)})
    assert allclose(jac["y_1"]["x_1"], array([[1.0], [-1.0]]))
    assert allclose(jac["y_1"]["x_2"], array([[1.0, 1.0], [-1.0, -1.0]]))


@pytest.mark.parametrize("variable", ["x_1", "y_1"])
def test_predict_jacobian_failure(dataset_for_jacobian, variable):
    """Test predict Jacobian when the transformer uses a variable name."""
    expected = re.escape(
        "The Jacobian of regression models cannot be computed "
        "when the transformed quantities are variables; "
        "please transform the whole group 'inputs' or 'outputs' "
        "or do not use data transformation."
    )
    ml_algo = LinearRegressor(
        dataset_for_jacobian, transformer={variable: "MinMaxScaler"}
    )
    ml_algo.learn()
    with pytest.raises(NotImplementedError, match=expected):
        ml_algo.predict_jacobian({"x_1": zeros(1), "x_2": zeros(2)})
