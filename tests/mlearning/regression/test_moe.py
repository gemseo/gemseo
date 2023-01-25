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
"""Test mixture of experts regression module."""
from __future__ import annotations

import pytest
from gemseo.core.dataset import Dataset
from gemseo.mlearning.api import import_regression_model
from gemseo.mlearning.classification.random_forest import RandomForestClassifier
from gemseo.mlearning.cluster.kmeans import KMeans
from gemseo.mlearning.regression.linreg import LinearRegressor
from gemseo.mlearning.regression.moe import MOERegressor
from gemseo.mlearning.transform.scaler.min_max_scaler import MinMaxScaler
from gemseo.mlearning.transform.scaler.scaler import Scaler
from numpy import allclose
from numpy import array
from numpy import hstack
from numpy import linspace
from numpy import meshgrid
from numpy import newaxis
from numpy import ones_like

ROOT_LEARNING_SIZE = 6
LEARNING_SIZE = ROOT_LEARNING_SIZE**2

INPUT_VALUE = {"x_1": array([1.0]), "x_2": array([2.0])}
ARRAY_INPUT_VALUE = array([1.0, 2.0])
INPUT_VALUES = {
    "x_1": array([[1.0], [0.0], [-1.0], [0.5], [0.1], [1.0]]),
    "x_2": array([[2.0], [0.0], [1.0], [-0.7], [0.4], [0.5]]),
}


ATOL = 1e-5
RTOL = 1e-5


@pytest.fixture
def dataset() -> Dataset:
    """The dataset used to train the regression algorithms."""
    x_1 = linspace(0, 1, ROOT_LEARNING_SIZE)
    x_2 = linspace(0, 1, ROOT_LEARNING_SIZE)
    grid_x_1, grid_x_2 = meshgrid(x_1, x_2)
    x_1 = grid_x_1.flatten()[:, newaxis]
    x_2 = grid_x_2.flatten()[:, newaxis]
    y_1 = (1 - 2 * x_1 + 3 * x_2) - 4 * (1 - x_1 - x_2) * (1 - x_1 - x_2 < 0)
    z_1 = x_1 + x_2
    z_2 = ones_like(z_1)
    z_2[0, 0] = 0
    data = hstack([x_1, x_2, y_1, z_1, z_2])
    variables = ["x_1", "x_2", "y", "z"]
    sizes = {"x_1": 1, "x_2": 1, "y": 1, "z": 2}
    groups = {"x_1": "inputs", "x_2": "inputs", "y": "outputs", "z": "outputs"}
    tmp = Dataset("dataset_name")
    tmp.set_from_array(data, variables, sizes, groups)
    return tmp


@pytest.fixture
def model(dataset) -> MOERegressor:
    """A trained MOERegressor."""
    moe = MOERegressor(dataset)
    moe.set_clusterer("KMeans", n_clusters=2)
    moe.learn()
    return moe


@pytest.fixture
def model_soft(dataset) -> MOERegressor:
    """A trained MOERegressor with soft classification."""
    moe = MOERegressor(dataset, hard=False)
    moe.set_clusterer("KMeans", n_clusters=2)
    moe.learn()
    return moe


@pytest.fixture
def model_with_transform(dataset) -> MOERegressor:
    """A trained MOERegressor with inputs and outputs scaling."""
    moe = MOERegressor(
        dataset, transformer={"inputs": MinMaxScaler(), "outputs": MinMaxScaler()}
    )
    moe.set_clusterer("KMeans", n_clusters=2)
    moe.set_regressor(
        "LinearRegressor",
        transformer={"inputs": Scaler(offset=5), "outputs": Scaler(coefficient=-1)},
    )
    moe.learn()
    return moe


def test_constructor(dataset):
    """Test construction."""
    moe = MOERegressor(dataset)
    assert moe.cluster_algo is not None
    assert moe.classif_algo is not None
    assert moe.regress_algo is not None


def test_learn(dataset):
    """Test learn."""
    moe = MOERegressor(dataset)
    moe.learn()
    assert moe.clusterer is not None
    assert moe.classifier is not None
    assert moe.regress_models is not None
    for label in moe.clusterer.labels:
        assert label in moe.labels
    assert len(moe.labels) == len(dataset)


def test_set_algos(dataset):
    """Test learn."""
    moe = MOERegressor(dataset)
    moe.set_classifier("RandomForestClassifier")
    moe.set_clusterer("KMeans", n_clusters=3)
    moe.set_regressor("LinearRegressor", fit_intercept=False)
    moe.learn()
    assert isinstance(moe.clusterer, KMeans)
    assert isinstance(moe.classifier, RandomForestClassifier)
    for local_model in moe.regress_models:
        assert isinstance(local_model, LinearRegressor)


def test_predict_class(model, model_with_transform):
    """Test class prediction."""
    prediction = model.predict_class(INPUT_VALUE)
    assert isinstance(prediction, dict)
    assert prediction["labels"].shape == (1,)
    assert model.predict_class(ARRAY_INPUT_VALUE) == prediction["labels"]

    prediction = model.predict_class(INPUT_VALUES)
    assert isinstance(prediction, dict)
    assert prediction["labels"].shape == (6, 1)

    prediction = model_with_transform.predict_class(INPUT_VALUES)
    assert isinstance(prediction, dict)
    assert prediction["labels"].shape == (6, 1)


def test_predict_local_model(model):
    """Test prediction of individual regression model."""
    prediction = model.predict_local_model(INPUT_VALUE, 0)
    option_1 = allclose(prediction["y"][0], 11.22893081, atol=ATOL, rtol=RTOL)
    prediction = model.predict_local_model(INPUT_VALUE, 1)
    option_2 = allclose(prediction["y"][0], 11.22893081, atol=ATOL, rtol=RTOL)
    assert option_1 or option_2
    predictions = model.predict_local_model(INPUT_VALUES, 0)
    assert isinstance(predictions, dict)
    assert "y" in predictions
    assert predictions["y"].shape == (6, 1)


def test_local_model_transform(model_with_transform):
    """Test prediction of individual regression model."""
    prediction = model_with_transform.predict_local_model(INPUT_VALUES, 0)
    assert isinstance(prediction, dict)
    assert "y" in prediction
    assert prediction["y"].shape == (6, 1)


def test_predict(model):
    """Test prediction."""
    prediction = model.predict(INPUT_VALUES)
    assert isinstance(prediction, dict)
    assert "y" in prediction
    assert prediction["y"].shape == (6, 1)


def test_predict_jacobian(model):
    """Test jacobian prediction."""
    jac = model.predict_jacobian(INPUT_VALUES)
    assert isinstance(jac, dict)
    assert jac["y"]["x_1"].shape == (6, 1, 1)
    assert jac["y"]["x_2"].shape == (6, 1, 1)


def test_predict_jacobian_soft(model_soft):
    """Test predict jacobian soft."""
    with pytest.raises(NotImplementedError):
        model_soft.predict_jacobian(INPUT_VALUES)


def test_save_and_load(model, tmp_wd):
    """Test save and load."""
    dirname = model.save()
    imported_model = import_regression_model(dirname)
    input_value = {"x_1": array([1.0]), "x_2": array([2.0])}
    out1 = model.predict(input_value)
    out2 = imported_model.predict(input_value)
    for name, value in out1.items():
        assert allclose(value, out2[name], 1e-3)


def test_str(model):
    """Test str representation."""
    repres = str(model)
    assert "MOERegressor" in repres
    assert "KMeans" in repres
    assert "KNNClassifier" in repres
    assert "Local model 0" in repres
    assert "Local model 1" in repres
    assert "Local model 2" not in repres


def test_moe_with_candidates(dataset):
    moe = MOERegressor(dataset)

    assert not moe.cluster_cands
    assert not moe.regress_cands
    assert not moe.classif_cands

    moe.add_clusterer_candidate("GaussianMixture", n_components=[5])
    assert len(moe.cluster_cands) == 1

    moe.add_classifier_candidate("SVMClassifier", kernel=["rbf"])
    assert len(moe.classif_cands) == 1

    moe.add_regressor_candidate("PolynomialRegressor", degree=[2])
    assert len(moe.regress_cands) == 1

    moe.learn()
    assert moe.classifier.__class__.__name__ == "SVMClassifier"
    assert moe.clusterer.__class__.__name__ == "GaussianMixture"
    for regression_model in moe.regress_models:
        assert regression_model.__class__.__name__ == "PolynomialRegressor"
