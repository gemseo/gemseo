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
from numpy import array

from gemseo.datasets.dataset import Dataset
from gemseo.mlearning.clustering.algos.base_clusterer import BaseClusterer
from gemseo.mlearning.clustering.algos.base_predictive_clusterer import (
    BasePredictiveClusterer,
)
from gemseo.mlearning.clustering.quality.base_clusterer_quality import (
    BaseClustererQuality,
)
from gemseo.mlearning.clustering.quality.base_predictive_clusterer_quality import (
    BasePredictiveClustererQuality,
)
from gemseo.mlearning.clustering.quality.factory import ClustererQualityFactory
from gemseo.utils.testing.helpers import concretize_classes


@pytest.fixture
def learning_data() -> Dataset:
    """The dataset used to train the clustering algorithms."""
    return Dataset.from_array(array([[1, 0], [2, 0], [3, 1], [4, 1]]), ["x", "y"])


@pytest.fixture
def test_data() -> Dataset:
    """The dataset used to test the performance clustering algorithms."""
    return Dataset.from_array(array([[1, 0.5]]), ["x", "y"])


class NewAlgo(BaseClusterer):
    def _fit(self, data) -> None:
        self.labels = data[:, 1]


class NewPredictiveAlgo(BasePredictiveClusterer):
    def _fit(self, data) -> None:
        self.labels = data[:, 1]

    def _predict(self, data):
        return array([int(value <= 2.0) for value in data[:, 0]])

    def _predict_proba_soft(self, data):
        return array([[0.2, 0.6, 0.2] for _ in data])


class NewClustererQuality(BaseClustererQuality):
    def _compute_measure(self, data, labels, multioutput=True) -> float:
        return 1.0


class NewPredictiveClustererQuality(BasePredictiveClustererQuality):
    def _compute_measure(self, data, labels, multioutput=True) -> float:
        return 1.0


@pytest.mark.parametrize("train", [False, True])
@pytest.mark.parametrize("samples", [(), [1, 2, 3]])
@pytest.mark.parametrize("multioutput", [True, False])
def test_compute_learning_measure(learning_data, train, samples, multioutput) -> None:
    algo = NewAlgo(learning_data)
    if train:
        algo.learn()

    with concretize_classes(NewClustererQuality):
        assert (
            NewClustererQuality(algo).compute_learning_measure(samples, multioutput)
            == 1.0
        )


@pytest.mark.parametrize("train", [False, True])
@pytest.mark.parametrize("samples", [(), [1, 2, 3]])
@pytest.mark.parametrize("multioutput", [True, False])
def test_compute_test_measure(
    learning_data, test_data, train, samples, multioutput
) -> None:
    algo = NewPredictiveAlgo(learning_data)
    if train:
        algo.learn()
    assert (
        NewPredictiveClustererQuality(algo).compute_test_measure(
            test_data, samples, multioutput
        )
        == 1.0
    )


@pytest.mark.parametrize("n_replicates", [1, 2])
@pytest.mark.parametrize("samples", [(), [1, 2, 3]])
@pytest.mark.parametrize("multioutput", [True, False])
def test_compute_bootstrap_measure(
    learning_data, n_replicates, samples, multioutput
) -> None:
    algo = NewPredictiveAlgo(learning_data)
    assert (
        NewPredictiveClustererQuality(algo).compute_bootstrap_measure(
            n_replicates, samples, multioutput
        )
        == 1.0
    )


@pytest.mark.parametrize("n_folds", [2, 3])
@pytest.mark.parametrize("samples", [(), [0, 1, 2]])
@pytest.mark.parametrize("multioutput", [True, False])
def test_compute_cross_validation_measure(
    learning_data, n_folds, samples, multioutput
) -> None:
    algo = NewPredictiveAlgo(learning_data)
    assert (
        NewPredictiveClustererQuality(algo).compute_cross_validation_measure(
            n_folds, samples, multioutput
        )
        == 1.0
    )


def test_factory():
    """Check ClustererQualityFactory."""
    assert ClustererQualityFactory().is_available("SilhouetteMeasure")
