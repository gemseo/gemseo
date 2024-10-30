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
"""Test high-level functions for machine learning."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from numpy import arange
from numpy import array
from numpy import atleast_2d
from numpy import hstack
from numpy import ndarray

from gemseo import create_dataset
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.mlearning import create_classification_model
from gemseo.mlearning import create_clustering_model
from gemseo.mlearning import create_mlearning_model
from gemseo.mlearning import create_regression_model
from gemseo.mlearning import get_classification_models
from gemseo.mlearning import get_classification_options
from gemseo.mlearning import get_clustering_models
from gemseo.mlearning import get_clustering_options
from gemseo.mlearning import get_mlearning_models
from gemseo.mlearning import get_mlearning_options
from gemseo.mlearning import get_regression_models
from gemseo.mlearning import get_regression_options
from gemseo.mlearning.transformers.scaler.min_max_scaler import MinMaxScaler
from gemseo.scenarios.doe_scenario import DOEScenario

if TYPE_CHECKING:
    from gemseo.datasets.dataset import Dataset

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
    discipline.set_cache(discipline.CacheType.MEMORY_FULL)
    design_space = DesignSpace()
    design_space.add_variable("x_1", lower_bound=0.0, upper_bound=1.0)
    design_space.add_variable("x_2", lower_bound=0.0, upper_bound=1.0)
    scenario = DOEScenario(
        [discipline], "y_1", design_space, formulation_name="DisciplinaryOpt"
    )
    scenario.execute(algo_name="PYDOE_FULLFACT", n_samples=LEARNING_SIZE)
    return discipline.cache.to_dataset("dataset_name")


@pytest.fixture
def classification_data() -> tuple[ndarray, list[str], dict[str, str]]:
    """The dataset used to train the classification algorithms."""
    data = array([
        [0, 0],
        [0, 1],
        [0, 2],
        [1, 0],
        [1, 1],
        [1, 2],
        [2, 0],
        [2, 1],
        [2, 2],
    ])
    data = hstack((data, atleast_2d(arange(9)).T))
    variables = ["x_1", "x_2", "class"]
    groups = {"class": "outputs", "x_1": "inputs", "x_2": "inputs"}
    return data, variables, groups


@pytest.fixture
def cluster_data() -> tuple[ndarray, list[str]]:
    """The dataset used to train the clustering algorithms."""
    data = array([
        [0, 0],
        [0, 1],
        [0, 2],
        [1, 0],
        [1, 1],
        [1, 2],
        [2, 0],
        [2, 1],
        [2, 2],
    ])
    return data, ["x_1", "x_2"]


def test_get_mlearning_models() -> None:
    """Test that available ML models are found."""
    available_models = get_mlearning_models()
    for regression_model in AVAILABLE_REGRESSION_MODELS:
        assert regression_model in available_models
    for classification_model in AVAILABLE_CLASSIFICATION_MODELS:
        assert classification_model in available_models
    for clustering_model in AVAILABLE_CLUSTERING_MODELS:
        assert clustering_model in available_models
    assert "Dummy" not in available_models


def test_get_regression_models() -> None:
    """Test that available regression models are found."""
    available_models = get_regression_models()
    for regression_model in AVAILABLE_REGRESSION_MODELS:
        assert regression_model in available_models
    assert "Dummy" not in available_models


def test_get_classification_models() -> None:
    """Test that available classification models are found."""
    available_models = get_classification_models()
    for classification_model in AVAILABLE_CLASSIFICATION_MODELS:
        assert classification_model in available_models
    assert "Dummy" not in available_models


def test_get_clustering_models() -> None:
    """Test that available clustering models are found."""
    available_models = get_clustering_models()
    for clustering_model in AVAILABLE_CLUSTERING_MODELS:
        assert clustering_model in available_models
    assert "Dummy" not in available_models


def test_create_mlearning_model(dataset, classification_data, cluster_data) -> None:
    """Test creation of model."""
    model = create_mlearning_model("LinearRegressor", dataset)
    assert model.algo is not None
    data, variables, groups = classification_data
    dataset = create_dataset(
        "dataset_name",
        data,
        variables,
        variable_names_to_group_names=groups,
        class_name="IODataset",
    )
    model = create_classification_model("KNNClassifier", dataset)
    assert model.algo is not None
    data, variables = cluster_data
    dataset = create_dataset("dataset_name", data, variables)
    model = create_clustering_model("KMeans", dataset, n_clusters=data.shape[0])
    assert model.algo is not None


def test_create_regression_model(dataset) -> None:
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


def test_create_classification_model(classification_data) -> None:
    """Test creation of classification model."""
    data, variables, groups = classification_data
    dataset = create_dataset(
        "dataset_name",
        data,
        variables,
        variable_names_to_group_names=groups,
        class_name="IODataset",
    )
    model = create_classification_model("KNNClassifier", dataset)
    assert model.algo is not None


def test_create_clustering_model(cluster_data) -> None:
    """Test creation of clustering model."""
    data, variables = cluster_data
    dataset = create_dataset("dataset_name", data, variables)
    model = create_clustering_model("KMeans", dataset, n_clusters=data.shape[0])
    assert model.algo is not None


def test_get_mlearning_options() -> None:
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


def test_get_regression_options() -> None:
    """Test correct retrieval of regression model options."""
    properties = get_regression_options("LinearRegressor")["properties"]
    assert "fit_intercept" in properties
    assert "Dummy" not in properties


def test_get_classification_options() -> None:
    """Test correct retrieval of classification model options."""
    properties = get_classification_options("KNNClassifier")["properties"]
    assert "n_neighbors" in properties
    assert "Dummy" not in properties


def test_get_clustering_model_options() -> None:
    """Test correct retrieval of clustering model options."""
    properties = get_clustering_options("KMeans")["properties"]
    assert "n_clusters" in properties
    assert "Dummy" not in properties


@pytest.mark.parametrize(
    ("get_options", "class_name"),
    [
        (get_clustering_options, "KMeans"),
        (get_classification_options, "KNNClassifier"),
        (get_regression_options, "LinearRegressor"),
        (get_mlearning_options, "LinearRegressor"),
    ],
)
def test_get_options_output_json(get_options, class_name) -> None:
    """Check output_json argument of get_xxxx_options."""
    options = get_options(class_name, pretty_print=False, output_json=True)
    assert isinstance(options, str)
    assert class_name in options
