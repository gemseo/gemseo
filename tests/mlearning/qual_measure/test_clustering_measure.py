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
"""Test clustering measure module."""
from __future__ import annotations

import pytest
from gemseo.core.dataset import Dataset
from gemseo.mlearning.cluster.cluster import MLClusteringAlgo
from gemseo.mlearning.cluster.cluster import MLPredictiveClusteringAlgo
from gemseo.mlearning.qual_measure.cluster_measure import MLClusteringMeasure
from gemseo.mlearning.qual_measure.cluster_measure import MLPredictiveClusteringMeasure
from gemseo.utils.pytest_conftest import concretize_classes
from numpy import array


@pytest.fixture
def learning_data() -> Dataset:
    """The dataset used to train the clustering algorithms."""
    dataset = Dataset()
    dataset.set_from_array(array([[1, 0], [2, 0], [3, 1], [4, 1]]), ["x", "y"])
    return dataset


@pytest.fixture
def test_data() -> Dataset:
    """The dataset used to test the performance clustering algorithms."""
    dataset = Dataset()
    dataset.set_from_array(array([[1, 0.5]]), ["x", "y"])
    return dataset


class NewAlgo(MLClusteringAlgo):
    def _fit(self, data):
        self.labels = data[:, 1]


class NewPredictiveAlgo(MLPredictiveClusteringAlgo):
    def _fit(self, data):
        self.labels = data[:, 1]

    def _predict(self, data):
        return array([int(value <= 2.0) for value in data[:, 0]])

    def _predict_proba_soft(self, data):
        return array([[0.2, 0.6, 0.2] for _ in data])


class NewMLClusteringMeasure(MLClusteringMeasure):
    def _compute_measure(self, data, labels, multioutput=True):
        return 1.0


class NewMLPredictiveClusteringMeasure(MLPredictiveClusteringMeasure):
    def _compute_measure(self, data, labels, multioutput=True):
        return 1.0


@pytest.mark.parametrize("train", [False, True])
@pytest.mark.parametrize("samples", [None, [1, 2, 3]])
@pytest.mark.parametrize("multioutput", [True, False])
def test_evaluate_learn(learning_data, train, samples, multioutput):
    algo = NewAlgo(learning_data)
    if train:
        algo.learn()

    with concretize_classes(NewMLClusteringMeasure):
        assert NewMLClusteringMeasure(algo).evaluate_learn(samples, multioutput) == 1.0


@pytest.mark.parametrize("train", [False, True])
@pytest.mark.parametrize("samples", [None, [1, 2, 3]])
@pytest.mark.parametrize("multioutput", [True, False])
def test_evaluate_test(learning_data, test_data, train, samples, multioutput):
    algo = NewPredictiveAlgo(learning_data)
    if train:
        algo.learn()
    assert (
        NewMLPredictiveClusteringMeasure(algo).evaluate_test(
            test_data, samples, multioutput
        )
        == 1.0
    )


@pytest.mark.parametrize("n_replicates", [1, 2])
@pytest.mark.parametrize("samples", [None, [1, 2, 3]])
@pytest.mark.parametrize("multioutput", [True, False])
def test_evaluate_bootstrap(learning_data, n_replicates, samples, multioutput):
    algo = NewPredictiveAlgo(learning_data)
    assert (
        NewMLPredictiveClusteringMeasure(algo).evaluate_bootstrap(
            n_replicates, samples, multioutput
        )
        == 1.0
    )


@pytest.mark.parametrize("n_folds", [2, 3])
@pytest.mark.parametrize("samples", [None, [0, 1, 2]])
@pytest.mark.parametrize("multioutput", [True, False])
def test_evaluate_kfolds(learning_data, n_folds, samples, multioutput):
    algo = NewPredictiveAlgo(learning_data)
    assert (
        NewMLPredictiveClusteringMeasure(algo).evaluate_kfolds(
            n_folds, samples, multioutput
        )
        == 1.0
    )
