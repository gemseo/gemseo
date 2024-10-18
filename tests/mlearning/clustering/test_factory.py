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
"""Unit test for ClustererFactory class in gemseo.mlearning.clustering.factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from gemseo.mlearning.clustering.algos.factory import ClustererFactory
from gemseo.mlearning.clustering.algos.kmeans import KMeans
from gemseo.problems.dataset.iris import create_iris_dataset

if TYPE_CHECKING:
    from gemseo.datasets.io_dataset import IODataset

N_CLUSTERS = 3


@pytest.fixture
def dataset() -> IODataset:
    """The dataset used to train the clustering algorithms."""
    return create_iris_dataset(as_io=True, as_numeric=True)


def test_constructor() -> None:
    """Test ClustererFactory constructor."""
    factory = ClustererFactory()
    # plugins may add classes
    assert set(factory.class_names) <= {
        "GaussianMixture",
        "KMeans",
        "BasePredictiveClusterer",
    }


def test_create(dataset) -> None:
    """Test the creation of a model from data."""
    factory = ClustererFactory()
    assert isinstance(
        factory.create("KMeans", data=dataset, n_clusters=N_CLUSTERS), KMeans
    )


def test_is_available() -> None:
    """Test the existence of a clustering model."""
    factory = ClustererFactory()
    assert factory.is_available("KMeans")
    assert not factory.is_available("LinearRegressor")
