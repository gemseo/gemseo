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
#        :author: Syver Doving Agdestein
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Unit test for ClusteringModelFactory class in gemseo.mlearning.clustering.factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from gemseo.mlearning.clustering.factory import ClusteringModelFactory
from gemseo.problems.dataset.iris import create_iris_dataset

if TYPE_CHECKING:
    from gemseo.datasets.io_dataset import IODataset

N_CLUSTERS = 3


@pytest.fixture()
def dataset() -> IODataset:
    """The dataset used to train the clustering algorithms."""
    return create_iris_dataset(as_io=True, as_numeric=True)


def test_constructor():
    """Test ClusteringModelFactory constructor."""
    factory = ClusteringModelFactory()
    # plugins may add classes
    assert set(factory.models) <= {
        "GaussianMixture",
        "KMeans",
        "MLPredictiveClusteringAlgo",
    }


def test_create(dataset):
    """Test the creation of a model from data."""
    factory = ClusteringModelFactory()
    kmeans = factory.create("KMeans", data=dataset, n_clusters=N_CLUSTERS)
    assert hasattr(kmeans, "parameters")


def test_load(dataset, tmp_wd):
    """Test the loading of a model from data."""
    factory = ClusteringModelFactory()
    kmeans = factory.create("KMeans", data=dataset, n_clusters=N_CLUSTERS)
    kmeans.learn()
    dirname = kmeans.to_pickle()
    loaded_kmeans = factory.load(dirname)
    assert hasattr(loaded_kmeans, "parameters")


def test_available_clustering_models():
    """Test the getter of available clustering models."""
    factory = ClusteringModelFactory()
    assert "KMeans" in factory.models
    assert "LinearRegressor" not in factory.models


def test_is_available():
    """Test the existence of a clustering model."""
    factory = ClusteringModelFactory()
    assert factory.is_available("KMeans")
    assert not factory.is_available("Dummy")
