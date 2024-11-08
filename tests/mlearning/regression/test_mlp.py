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
from __future__ import annotations

import pytest
import sklearn.neural_network
from numpy import array

from gemseo.mlearning.regression.algos.mlp import MLPRegressor


def test_init(dataset):
    """Check that the wrapped algorithm is MLPRegressor from sklearn.

    Check also the default number of neurons per hidden layer.
    """
    mlp = MLPRegressor(dataset)
    assert mlp.SHORT_ALGO_NAME == "MLP"
    assert mlp.LIBRARY == "scikit-learn"
    algo = mlp.algo
    assert isinstance(algo, sklearn.neural_network.MLPRegressor)
    assert algo.hidden_layer_sizes == (100,)


def test_init_hidden_layer_sizes(dataset):
    """Check that the hidden layer sizes can be changed."""
    algo = MLPRegressor(dataset, hidden_layer_sizes=(3, 2)).algo
    assert algo.hidden_layer_sizes == (3, 2)


def test_init_parameter(dataset):
    """Check that a sklearn parameter can be changed."""
    algo = MLPRegressor(dataset, parameters={"activation": "identity"}).algo
    assert algo.activation == "identity"


def test_fit(dataset, input_data, output_data):
    """Check the learning stage."""
    mlp = MLPRegressor(dataset)
    assert not hasattr(mlp, "coefs_")
    mlp._fit(input_data, output_data)
    assert len(mlp.algo.coefs_) == 2


@pytest.mark.parametrize("output_name", ["rosen", "rosen2"])
def test_predict(dataset_2, output_name):
    """Check the prediction stage."""
    algo = MLPRegressor(dataset_2, output_names=[output_name])
    algo.learn()
    input_data = array([[1.0, 1.0]])
    assert algo._predict(input_data).shape == (1, algo.output_dimension)
    with pytest.raises(NotImplementedError):
        algo.predict_jacobian(input_data)
