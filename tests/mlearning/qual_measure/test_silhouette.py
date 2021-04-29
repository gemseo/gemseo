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
#        :author: Syver Doving Agdestein
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Test silhouette measure."""
from __future__ import absolute_import, division, unicode_literals

import pytest
from numpy import arange

from gemseo.core.dataset import Dataset
from gemseo.mlearning.cluster.kmeans import KMeans
from gemseo.mlearning.core.ml_algo import MLAlgo
from gemseo.mlearning.qual_measure.silhouette import SilhouetteMeasure


@pytest.fixture
def dataset():
    """Data points."""
    data = arange(60).reshape((20, 3))
    dataset_ = Dataset()
    dataset_.add_variable("x", data, Dataset.PARAMETER_GROUP)
    return dataset_


@pytest.fixture
def dataset_test():
    """Data points."""
    data = arange(30).reshape((10, 3))
    dataset_ = Dataset()
    dataset_.add_variable("x", data, Dataset.PARAMETER_GROUP)
    return dataset_


def test_constructor(dataset):
    """Test construction."""
    algo = MLAlgo(dataset)
    measure = SilhouetteMeasure(algo)
    assert measure.algo is not None
    assert measure.algo.learning_set is dataset


def test_evaluate_learn(dataset):
    """Test evaluate learn method."""
    algo = KMeans(dataset, n_clusters=3)
    measure = SilhouetteMeasure(algo)
    quality = measure.evaluate("learn", multioutput=False)
    assert quality > 0


def test_evaluate_test(dataset, dataset_test):
    """Test evaluate test method."""
    algo = KMeans(dataset, n_clusters=3)
    measure = SilhouetteMeasure(algo)
    with pytest.raises(NotImplementedError):
        measure.evaluate("test", test_data=dataset_test, multioutput=False)


def test_evaluate_loo(dataset):
    """Test evaluate leave one out method."""
    algo = KMeans(dataset, n_clusters=3)
    measure = SilhouetteMeasure(algo)
    with pytest.raises(NotImplementedError):
        quality = measure.evaluate("loo", multioutput=False)
        assert quality > 0


def test_evaluate_kfolds(dataset):
    """Test evaluate k-folds method."""
    algo = KMeans(dataset, n_clusters=3)
    measure = SilhouetteMeasure(algo)
    with pytest.raises(NotImplementedError):
        quality = measure.evaluate("kfolds", multioutput=False)
        assert quality > 0


def test_evaluate_bootstrap(dataset):
    """Test evaluate bootstrap method."""
    algo = KMeans(dataset, n_clusters=3)
    measure = SilhouetteMeasure(algo)
    with pytest.raises(NotImplementedError):
        quality = measure.evaluate("bootstrap", multioutput=False)
        assert quality > 0
