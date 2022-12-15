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
"""Test random forest classification module."""
from __future__ import annotations

import pytest
from gemseo.core.dataset import Dataset
from gemseo.mlearning.api import import_classification_model
from gemseo.mlearning.classification.random_forest import RandomForestClassifier
from gemseo.mlearning.transform.scaler.min_max_scaler import MinMaxScaler
from numpy import allclose
from numpy import array
from numpy import array_equal
from numpy import linspace
from numpy import ndarray
from numpy import zeros
from numpy.random import permutation
from numpy.random import seed

seed(12345)

N_INPUTS = 2
N_OUTPUTS = 3
N_CLASSES = 4

INPUT_VALUE = {"x_1": array([1.0]), "x_2": array([1.0])}

INPUT_VALUES = {
    "x_1": array([[1.0], [0.0], [1.0], [0.0], [0.5]]),
    "x_2": array([[1.0], [0.0], [0.0], [1.0], [0.5]]),
}


@pytest.fixture
def dataset():
    """The dataset used to train the RandomForestClassifier."""
    input_data = linspace(0, 1, 20).reshape((10, 2))
    output_data = zeros((10, 3))
    output_data[::4, 0] = 1
    output_data[1::4, 0] = 2
    output_data[2::4, 0] = 3
    output_data[:, 1] = permutation(output_data[:, 0])
    output_data[:, 2] = permutation(output_data[:, 0])
    dataset_ = Dataset()
    dataset_.add_group(
        Dataset.INPUT_GROUP, input_data, ["x_1", "x_2"], {"x_1": 1, "x_2": 1}
    )
    dataset_.add_group(
        Dataset.OUTPUT_GROUP, output_data, ["y_1", "y_2"], {"y_1": 1, "y_2": 2}
    )
    return dataset_


@pytest.fixture
def model_1d(dataset) -> RandomForestClassifier:
    """A trained RandomForestClassifier with y_1 as single output."""
    algo = RandomForestClassifier(dataset, output_names=["y_1"])
    algo.learn()
    return algo


@pytest.fixture
def model(dataset) -> RandomForestClassifier:
    """A trained KNNClassifier with two outputs, y_1 and y_2."""
    algo = RandomForestClassifier(dataset)
    algo.learn()
    return algo


@pytest.fixture
def model_with_transform(dataset) -> RandomForestClassifier:
    """A trained KNNClassifier using input scaling."""
    algo = RandomForestClassifier(dataset, transformer={"inputs": MinMaxScaler()})
    algo.learn()
    return algo


def test_constructor(dataset):
    """Test construction."""
    algo = RandomForestClassifier(dataset)
    assert algo.algo is not None
    assert algo.SHORT_ALGO_NAME == "RF"
    assert algo.LIBRARY == "scikit-learn"


def test_learn(dataset):
    """Test learn."""
    algo = RandomForestClassifier(dataset)
    algo.learn()
    assert algo.algo is not None


def test_predict_1d(model_1d):
    """Test prediction."""
    prediction = model_1d.predict(INPUT_VALUE)
    predictions = model_1d.predict(INPUT_VALUES)

    assert isinstance(prediction, dict)
    assert isinstance(prediction["y_1"], ndarray)
    assert prediction["y_1"].shape == (1,)

    assert isinstance(predictions, dict)
    assert isinstance(predictions["y_1"], ndarray)
    assert predictions["y_1"].shape == (5, 1)


def test_predict(model):
    """Test prediction."""
    prediction = model.predict(INPUT_VALUE)
    predictions = model.predict(INPUT_VALUES)

    assert isinstance(prediction, dict)
    assert isinstance(prediction["y_1"], ndarray)
    assert isinstance(prediction["y_2"], ndarray)
    assert prediction["y_1"].shape == (1,)
    assert prediction["y_2"].shape == (2,)

    assert isinstance(predictions, dict)
    assert isinstance(predictions["y_1"], ndarray)
    assert isinstance(predictions["y_2"], ndarray)
    assert predictions["y_1"].shape == (5, 1)
    assert predictions["y_2"].shape == (5, 2)


def test_predict_with_transform(model_with_transform):
    """Test prediction."""
    prediction = model_with_transform.predict(INPUT_VALUE)
    predictions = model_with_transform.predict(INPUT_VALUES)

    assert isinstance(prediction, dict)
    assert isinstance(prediction["y_1"], ndarray)
    assert isinstance(prediction["y_2"], ndarray)
    assert prediction["y_1"].shape == (1,)
    assert prediction["y_2"].shape == (2,)

    assert isinstance(predictions, dict)
    assert isinstance(predictions["y_1"], ndarray)
    assert isinstance(predictions["y_2"], ndarray)

    assert predictions["y_1"].shape == (5, 1)
    assert predictions["y_2"].shape == (5, 2)


def test_predict_proba_1d(model_1d):
    """Test probability prediction."""
    for hard in [True, False]:
        proba = model_1d.predict_proba(INPUT_VALUE, hard)
        probas = model_1d.predict_proba(INPUT_VALUES, hard)
        assert isinstance(proba, dict)
        assert isinstance(probas, dict)
        assert isinstance(proba["y_1"], ndarray)
        assert isinstance(probas["y_1"], ndarray)
        assert proba["y_1"].shape == (4, 1)
        assert probas["y_1"].shape == (5, 4, 1)

        # Probas should add up to one
        assert allclose(proba["y_1"].sum(0), 1)
        assert allclose(probas["y_1"].sum(axis=1), 1)


def test_predict_proba(model):
    """Test probability prediction."""
    for hard in [True, False]:
        proba = model.predict_proba(INPUT_VALUE, hard)
        probas = model.predict_proba(INPUT_VALUES, hard)
        assert isinstance(proba, dict)
        assert isinstance(probas, dict)
        assert isinstance(proba["y_1"], ndarray)
        assert isinstance(proba["y_2"], ndarray)
        assert isinstance(probas["y_1"], ndarray)
        assert isinstance(probas["y_2"], ndarray)
        assert proba["y_1"].shape == (4, 1)
        assert proba["y_2"].shape == (4, 2)
        assert probas["y_1"].shape == (5, 4, 1)
        assert probas["y_2"].shape == (5, 4, 2)

        # Probas should add up to one
        assert allclose(proba["y_1"].sum(0), 1)
        assert allclose(proba["y_2"].sum(0), 1)
        assert allclose(probas["y_1"].sum(axis=1), 1)
        assert allclose(probas["y_2"].sum(axis=1), 1)


def test_predict_proba_transform(model_with_transform):
    """Test probability prediction."""
    for hard in [True, False]:
        proba = model_with_transform.predict_proba(INPUT_VALUE, hard)
        probas = model_with_transform.predict_proba(INPUT_VALUES, hard)
        assert isinstance(proba, dict)
        assert isinstance(probas, dict)
        assert isinstance(proba["y_1"], ndarray)
        assert isinstance(proba["y_2"], ndarray)
        assert isinstance(probas["y_1"], ndarray)
        assert isinstance(probas["y_2"], ndarray)
        assert proba["y_1"].shape == (4, 1)
        assert proba["y_2"].shape == (4, 2)
        assert probas["y_1"].shape == (5, 4, 1)
        assert probas["y_2"].shape == (5, 4, 2)

        # Probas should add up to one
        assert allclose(proba["y_1"].sum(0), 1)
        assert allclose(proba["y_2"].sum(0), 1)
        assert allclose(probas["y_1"].sum(axis=1), 1)
        assert allclose(probas["y_2"].sum(axis=1), 1)


def test_save_and_load(model, tmp_wd):
    """Test save and load."""
    dirname = model.save()
    imported_model = import_classification_model(dirname)
    out1 = model.predict(INPUT_VALUE)
    out2 = imported_model.predict(INPUT_VALUE)
    assert array_equal(out1["y_1"], out2["y_1"])
    assert array_equal(out1["y_2"], out2["y_2"])
