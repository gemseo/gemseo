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
from numpy import array
from numpy import hstack
from numpy import linspace
from numpy import meshgrid
from numpy import newaxis
from numpy import ones_like

from gemseo.datasets.io_dataset import IODataset
from gemseo.machine_learning.classification.models.knn_settings import (
    KNNClassifier_Settings,
)
from gemseo.machine_learning.classification.models.random_forest import (
    RandomForestClassifier,
)
from gemseo.machine_learning.classification.models.random_forest_settings import (
    RandomForestClassifier_Settings,
)
from gemseo.machine_learning.classification.models.svm_settings import (
    SVMClassifier_Settings,
)
from gemseo.machine_learning.clustering.models.gaussian_mixture_settings import (
    GaussianMixture_Settings,
)
from gemseo.machine_learning.clustering.models.kmeans import KMeans
from gemseo.machine_learning.clustering.models.kmeans_settings import KMeans_Settings
from gemseo.machine_learning.regression.models.linreg import LinearRegressor
from gemseo.machine_learning.regression.models.linreg_settings import (
    LinearRegressor_Settings,
)
from gemseo.machine_learning.regression.models.moe import MOERegressor
from gemseo.machine_learning.regression.models.moe_settings import MOERegressor_Settings
from gemseo.machine_learning.regression.models.polyreg_settings import (
    PolynomialRegressor_Settings,
)
from gemseo.machine_learning.transformers.scaler.min_max_scaler import MinMaxScaler
from gemseo.machine_learning.transformers.scaler.scaler import Scaler
from gemseo.utils.repr_html import REPR_HTML_WRAPPER

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
def dataset() -> IODataset:
    """The dataset used to train the regression models."""
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
    variable_name_to_n_components = {"x_1": 1, "x_2": 1, "y": 1, "z": 2}
    variable_name_to_group_name = {
        "x_1": "inputs",
        "x_2": "inputs",
        "y": "outputs",
        "z": "outputs",
    }
    tmp = IODataset.from_array(
        data, variables, variable_name_to_n_components, variable_name_to_group_name
    )
    tmp.name = "dataset_name"
    return tmp


@pytest.fixture
def model(dataset) -> MOERegressor:
    """A trained MOERegressor."""
    moe = MOERegressor(dataset)
    moe.set_clusterer(KMeans_Settings(n_clusters=2))
    moe.learn()
    return moe


@pytest.fixture
def model_soft(dataset) -> MOERegressor:
    """A trained MOERegressor with soft classification."""
    moe = MOERegressor(dataset, MOERegressor_Settings(hard=False))
    moe.set_clusterer(KMeans_Settings(n_clusters=2))
    moe.learn()
    return moe


@pytest.fixture
def model_with_transform(dataset) -> MOERegressor:
    """A trained MOERegressor with inputs and outputs scaling."""
    moe = MOERegressor(
        dataset,
        MOERegressor_Settings(
            transformer={"inputs": MinMaxScaler(), "outputs": MinMaxScaler()}
        ),
    )
    moe.set_clusterer(KMeans_Settings(n_clusters=2))
    moe.set_regressor(
        LinearRegressor_Settings(
            transformer={"inputs": Scaler(offset=5), "outputs": Scaler(coefficient=-1)}
        )
    )
    moe.learn()
    return moe


def test_constructor(dataset) -> None:
    """Test construction."""
    moe = MOERegressor(dataset)
    assert moe.clusterer_settings == KMeans_Settings()
    assert moe.classifier_settings == KNNClassifier_Settings()
    assert moe.regressor_settings == LinearRegressor_Settings()


def test_learn(dataset) -> None:
    """Test learn."""
    moe = MOERegressor(dataset)
    moe.learn()
    assert moe.clusterer is not None
    assert moe.classifier is not None
    assert moe.regressors is not None
    for label in moe.clusterer.labels:
        assert label in moe.labels
    assert len(moe.labels) == len(dataset)


def test_set_models(dataset) -> None:
    """Test learn."""
    moe = MOERegressor(dataset)
    moe.set_classifier(RandomForestClassifier_Settings())
    moe.set_clusterer(KMeans_Settings(n_clusters=3))
    moe.set_regressor(LinearRegressor_Settings(fit_intercept=False))
    moe.learn()
    assert isinstance(moe.clusterer, KMeans)
    assert isinstance(moe.classifier, RandomForestClassifier)
    for local_model in moe.regressors:
        assert isinstance(local_model, LinearRegressor)


def test_predict_class(model, model_with_transform) -> None:
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


@pytest.mark.parametrize(
    ("input_data", "shape"), [(INPUT_VALUE, (1,)), (INPUT_VALUES, (6, 1))]
)
@pytest.mark.parametrize("index", [0, 1])
def test_predict_local_model(model, input_data, index, shape) -> None:
    """Test prediction of individual regression model."""
    prediction = model.predict_local_model(input_data, index)
    assert prediction["y"].shape == shape


def test_local_model_transform(model_with_transform) -> None:
    """Test prediction of individual regression model."""
    prediction = model_with_transform.predict_local_model(INPUT_VALUES, 0)
    assert isinstance(prediction, dict)
    assert "y" in prediction
    assert prediction["y"].shape == (6, 1)


def test_predict(model) -> None:
    """Test prediction."""
    prediction = model.predict(INPUT_VALUES)
    assert isinstance(prediction, dict)
    assert "y" in prediction
    assert prediction["y"].shape == (6, 1)


def test_predict_jacobian(model) -> None:
    """Test jacobian prediction."""
    jac = model.predict_jacobian(INPUT_VALUES)
    assert isinstance(jac, dict)
    assert jac["y"]["x_1"].shape == (6, 1, 1)
    assert jac["y"]["x_2"].shape == (6, 1, 1)


def test_predict_jacobian_soft(model_soft) -> None:
    """Test predict jacobian soft."""
    with pytest.raises(NotImplementedError):
        model_soft.predict_jacobian(INPUT_VALUES)


def test_repr_str_(model) -> None:
    """Test string representations."""
    expected = """MOERegressor(hard=True, input_names=(), output_names=(), parameters={}, transformer={})
   built from 36 learning samples
   Clustering
      KMeans(n_clusters=2, parameters={}, random_state=0, transformer={}, var_names=())
   Classification
      KNNClassifier(input_names=(), n_neighbors=5, output_names=['labels'], parameters={}, transformer={})
   Regression
      Local model 0
         LinearRegressor(fit_intercept=True, input_names=(), l2_penalty_ratio=1.0, output_names=(), parameters={}, penalty_level=0.0, random_state=0, transformer={})
      Local model 1
         LinearRegressor(fit_intercept=True, input_names=(), l2_penalty_ratio=1.0, output_names=(), parameters={}, penalty_level=0.0, random_state=0, transformer={})"""  # noqa: E501
    assert repr(model) == str(model) == expected


def test_repr_html(model) -> None:
    """Check MOERegressor._repr_html."""
    assert model._repr_html_() == REPR_HTML_WRAPPER.format(
        "MOERegressor(hard=True, input_names=(), output_names=(), parameters={}, "
        "transformer={})<br/>"
        "<ul>"
        "<li>built from 36 learning samples</li>"
        "<li>Clustering"
        "<ul>"
        "<li>KMeans(n_clusters=2, parameters={}, random_state=0, transformer={}, "
        "var_names=())</li>"
        "</ul>"
        "</li>"
        "<li>Classification"
        "<ul>"
        "<li>KNNClassifier(input_names=(), n_neighbors=5, "
        "output_names=[&#x27;labels&#x27;], parameters={}, transformer={})</li>"
        "</ul>"
        "</li>"
        "<li>Regression"
        "<ul>"
        "<li>Local model 0"
        "<ul>"
        "<li>LinearRegressor(fit_intercept=True, input_names=(), l2_penalty_ratio=1.0, "
        "output_names=(), parameters={}, penalty_level=0.0, random_state=0, "
        "transformer={})</li>"
        "</ul>"
        "</li>"
        "<li>Local model 1"
        "<ul>"
        "<li>LinearRegressor(fit_intercept=True, input_names=(), l2_penalty_ratio=1.0, "
        "output_names=(), parameters={}, penalty_level=0.0, random_state=0, "
        "transformer={})</li>"
        "</ul>"
        "</li>"
        "</ul>"
        "</li>"
        "</ul>"
    )


def test_moe_with_candidates(dataset) -> None:
    moe = MOERegressor(dataset)

    assert not moe.clustering_candidates
    assert not moe.regression_candidates
    assert not moe.classification_candidates

    moe.add_clusterer_candidate(GaussianMixture_Settings(n_clusters=5))
    assert len(moe.clustering_candidates) == 1

    moe.add_classifier_candidate(SVMClassifier_Settings(kernel="rbf"))
    assert len(moe.classification_candidates) == 1

    moe.add_regressor_candidate(PolynomialRegressor_Settings(degree=2))
    assert len(moe.regression_candidates) == 1

    moe.learn()
    assert moe.classifier.__class__.__name__ == "SVMClassifier"
    assert moe.clusterer.__class__.__name__ == "GaussianMixture"
    for regression_model in moe.regressors:
        assert regression_model.__class__.__name__ == "PolynomialRegressor"
