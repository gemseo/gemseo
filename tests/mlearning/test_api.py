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
"""Test machine learning API."""
from __future__ import annotations

import pickle
from pathlib import Path

import pytest
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.api import create_dataset
from gemseo.core.dataset import Dataset
from gemseo.core.doe_scenario import DOEScenario
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.mlearning.api import create_classification_model
from gemseo.mlearning.api import create_clustering_model
from gemseo.mlearning.api import create_mlearning_model
from gemseo.mlearning.api import create_regression_model
from gemseo.mlearning.api import get_classification_models
from gemseo.mlearning.api import get_classification_options
from gemseo.mlearning.api import get_clustering_models
from gemseo.mlearning.api import get_clustering_options
from gemseo.mlearning.api import get_mlearning_models
from gemseo.mlearning.api import get_mlearning_options
from gemseo.mlearning.api import get_regression_models
from gemseo.mlearning.api import get_regression_options
from gemseo.mlearning.api import import_classification_model
from gemseo.mlearning.api import import_clustering_model
from gemseo.mlearning.api import import_mlearning_model
from gemseo.mlearning.api import import_regression_model
from gemseo.mlearning.core.ml_algo import MLAlgo
from gemseo.mlearning.regression.linreg import LinearRegressor
from gemseo.mlearning.transform.scaler.min_max_scaler import MinMaxScaler
from numpy import arange
from numpy import array
from numpy import atleast_2d
from numpy import hstack
from numpy import ndarray

LEARNING_SIZE = 9
AVAILABLE_REGRESSION_MODELS = [
    "LinearRegressor",
    "PolynomialRegressor",
    "GaussianProcessRegressor",
    "PCERegressor",
    "RBFRegressor",
    "MOERegressor",
]
AVAILABLE_CLASSIFICATION_MODELS = ["KNNClassifier", "RandomForestClassifier"]
AVAILABLE_CLUSTERING_MODELS = ["KMeans", "GaussianMixture"]


@pytest.fixture
def dataset() -> Dataset:
    """The dataset used to train the machine learning algorithms."""
    discipline = AnalyticDiscipline({"y_1": "1+2*x_1+3*x_2", "y_2": "-1-2*x_1-3*x_2"})
    discipline.set_cache_policy(discipline.MEMORY_FULL_CACHE)
    design_space = DesignSpace()
    design_space.add_variable("x_1", l_b=0.0, u_b=1.0)
    design_space.add_variable("x_2", l_b=0.0, u_b=1.0)
    scenario = DOEScenario([discipline], "DisciplinaryOpt", "y_1", design_space)
    scenario.execute({"algo": "fullfact", "n_samples": LEARNING_SIZE})
    return discipline.cache.export_to_dataset("dataset_name")


@pytest.fixture
def classification_data() -> tuple[ndarray, list[str], dict[str, str]]:
    """The dataset used to train the classification algorithms."""
    data = array(
        [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
    )
    data = hstack((data, atleast_2d(arange(9)).T))
    variables = ["x_1", "x_2", "class"]
    groups = {"class": "outputs", "x_1": "inputs", "x_2": "inputs"}
    return data, variables, groups


@pytest.fixture
def cluster_data() -> tuple[ndarray, list[str]]:
    """The dataset used to train the clustering algorithms."""
    data = array(
        [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
    )
    return data, ["x_1", "x_2"]


def test_get_mlearning_models():
    """Test that available ML models are found."""
    available_models = get_mlearning_models()
    for regression_model in AVAILABLE_REGRESSION_MODELS:
        assert regression_model in available_models
    for classification_model in AVAILABLE_CLASSIFICATION_MODELS:
        assert classification_model in available_models
    for clustering_model in AVAILABLE_CLUSTERING_MODELS:
        assert clustering_model in available_models
    assert "Dummy" not in available_models


def test_get_regression_models():
    """Test that available regression models are found."""
    available_models = get_regression_models()
    for regression_model in AVAILABLE_REGRESSION_MODELS:
        assert regression_model in available_models
    assert "Dummy" not in available_models


def test_get_classification_models():
    """Test that available classification models are found."""
    available_models = get_classification_models()
    for classification_model in AVAILABLE_CLASSIFICATION_MODELS:
        assert classification_model in available_models
    assert "Dummy" not in available_models


def test_get_clustering_models():
    """Test that available clustering models are found."""
    available_models = get_clustering_models()
    for clustering_model in AVAILABLE_CLUSTERING_MODELS:
        assert clustering_model in available_models
    assert "Dummy" not in available_models


def test_create_mlearning_model(dataset, classification_data, cluster_data):
    """Test creation of model."""
    model = create_mlearning_model("LinearRegressor", dataset)
    assert model.algo is not None
    data, variables, groups = classification_data
    dataset = create_dataset("dataset_name", data, variables, groups=groups)
    model = create_classification_model("KNNClassifier", dataset)
    assert model.algo is not None
    data, variables = cluster_data
    dataset = create_dataset("dataset_name", data, variables)
    model = create_clustering_model("KMeans", dataset, n_clusters=data.shape[0])
    assert model.algo is not None


def test_create_regression_model(dataset):
    """Test creation of regression model."""
    model = create_regression_model("LinearRegressor", dataset)
    assert model.algo is not None

    probability_space = ParameterSpace()
    probability_space.add_random_variable(
        "x_1", "OTUniformDistribution", minimum=0, maximum=1
    )
    probability_space.add_random_variable(
        "x_2", "OTUniformDistribution", minimum=0, maximum=1
    )
    model = create_regression_model(
        "PCERegressor",
        dataset,
        probability_space=probability_space,
        transformer={"inputs": MinMaxScaler()},
    )
    assert not model.transformer


def test_create_classification_model(classification_data):
    """Test creation of classification model."""
    data, variables, groups = classification_data
    dataset = create_dataset("dataset_name", data, variables, groups=groups)
    model = create_classification_model("KNNClassifier", dataset)
    assert model.algo is not None


def test_create_clustering_model(cluster_data):
    """Test creation of clustering model."""
    data, variables = cluster_data
    dataset = create_dataset("dataset_name", data, variables)
    model = create_clustering_model("KMeans", dataset, n_clusters=data.shape[0])
    assert model.algo is not None


def test_import_mlearning_model(dataset, classification_data, cluster_data, tmp_wd):
    """Test import of model."""
    model = create_mlearning_model("LinearRegressor", dataset)
    model.learn()
    dirname = model.save()
    loaded_model = import_mlearning_model(dirname)
    assert hasattr(loaded_model, "parameters")
    data, variables = cluster_data
    dataset = create_dataset("dataset_name", data, variables)
    model = create_mlearning_model("KMeans", dataset)
    model.learn()
    dirname = model.save()
    loaded_model = import_mlearning_model(dirname)
    assert hasattr(loaded_model, "parameters")
    data, variables, groups = classification_data
    dataset = create_dataset("dataset_name", data, variables, groups=groups)
    model = create_mlearning_model("RandomForestClassifier", dataset)
    model.learn()
    dirname = model.save()
    loaded_model = import_mlearning_model(dirname)
    assert hasattr(loaded_model, "parameters")


def test_import_regression_model(dataset, tmp_wd):
    """Test import of regression model."""
    model = create_regression_model("LinearRegressor", dataset)
    model.learn()
    dirname = model.save()
    loaded_model = import_regression_model(dirname)
    assert hasattr(loaded_model, "parameters")


def test_import_regression_model_with_old_class_name():
    """Test import of a regression model with an old class name.

    An instance of LinearRegression has been saved with GEMSEO 3.2.2;
    GEMSEO 3.0 renamed LinearRegression to LinearRegressor.

    This test checks the use of the mapping MLFactory.__OLD_TO_NEW_NAMES.
    """
    directory = Path(__file__).parent / "old_algo"
    loaded_model = import_regression_model(directory)
    assert isinstance(loaded_model, LinearRegressor)
    with (directory / MLAlgo.FILENAME).open("rb") as f:
        objects = pickle.load(f)

    assert objects.pop("algo_name") == "LinearRegression"


def test_import_classification_model(classification_data, tmp_wd):
    """Test import of classification model."""
    data, variables, groups = classification_data
    dataset = create_dataset("dataset_name", data, variables, groups=groups)
    model = create_classification_model("KNNClassifier", dataset)
    model.learn()
    dirname = model.save()
    loaded_model = import_classification_model(dirname)
    assert hasattr(loaded_model, "parameters")


def test_import_clustering_model(cluster_data, tmp_wd):
    """Test import of clustering model."""
    data, variables = cluster_data
    dataset = create_dataset("dataset_name", data, variables)
    model = create_clustering_model("KMeans", dataset, n_clusters=data.shape[0])
    model.learn()
    dirname = model.save()
    loaded_model = import_clustering_model(dirname)
    assert hasattr(loaded_model, "parameters")


def test_get_mlearning_options():
    """Test correct retrieval of model options."""
    properties = get_mlearning_options("LinearRegressor")["properties"]
    assert "fit_intercept" in properties
    assert "Dummy" not in properties
    properties = get_mlearning_options("KNNClassifier")["properties"]
    assert "n_neighbors" in properties
    assert "Dummy" not in properties
    properties = get_mlearning_options("KMeans")["properties"]
    assert "n_clusters" in properties
    assert "Dummy" not in properties


def test_get_regression_options():
    """Test correct retrieval of regression model options."""
    properties = get_regression_options("LinearRegressor")["properties"]
    assert "fit_intercept" in properties
    assert "Dummy" not in properties


def test_get_classification_options():
    """Test correct retrieval of classification model options."""
    properties = get_classification_options("KNNClassifier")["properties"]
    assert "n_neighbors" in properties
    assert "Dummy" not in properties


def test_get_clustering_model_options():
    """Test correct retrieval of clustering model options."""
    properties = get_clustering_options("KMeans")["properties"]
    assert "n_clusters" in properties
    assert "Dummy" not in properties
