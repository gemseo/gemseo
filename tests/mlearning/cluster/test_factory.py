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
#                           documentation
#        :author: Matthias De Lozzo
#        :author: Syver Doving Agdestein
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Unit test for ClusteringModelFactory class
in gemseo.mlearning.cluster.factory.
"""
from __future__ import absolute_import, division, unicode_literals

import pytest
from future import standard_library

from gemseo.mlearning.cluster.factory import ClusteringModelFactory
from gemseo.problems.dataset.iris import IrisDataset

standard_library.install_aliases()


N_CLUSTERS = 3


@pytest.fixture
def dataset():
    """ Dataset from Iris Dataset. """
    iris = IrisDataset()
    return iris


def test_constructor():
    """ Test ClusteringModelFactory constructor. """
    factory = ClusteringModelFactory()
    internal_modules_paths = factory.factory.internal_modules_paths
    assert "gemseo.mlearning.cluster" in internal_modules_paths


def test_create(dataset):
    """ Test the creation of a model from data. """
    factory = ClusteringModelFactory()
    kmeans = factory.create("KMeans", data=dataset, n_clusters=N_CLUSTERS)
    assert hasattr(kmeans, "parameters")


def test_load(dataset, tmp_path):
    """ Test the loading of a model from data. """
    factory = ClusteringModelFactory()
    kmeans = factory.create("KMeans", data=dataset, n_clusters=N_CLUSTERS)
    kmeans.learn()
    dirname = kmeans.save(path=str(tmp_path))
    loaded_kmeans = factory.load(dirname)
    assert hasattr(loaded_kmeans, "parameters")


def test_available_clustering_models():
    """ Test the getter of available clustering models. """
    factory = ClusteringModelFactory()
    assert "KMeans" in factory.models
    assert "LinearRegression" not in factory.models


def test_is_available():
    """ Test the existence of a clustering model. """
    factory = ClusteringModelFactory()
    assert factory.is_available("KMeans")
    assert not factory.is_available("Dummy")
