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
#                         documentation
#        :author: Syver Doving Agdestein
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Tests for K-means clustering model."""

from __future__ import annotations

import pytest
from numpy import allclose
from numpy import array
from numpy import integer
from numpy import ndarray
from numpy import vstack
from numpy.linalg import eigvals
from numpy.random import default_rng

from gemseo.datasets.io_dataset import IODataset
from gemseo.mlearning.clustering.algos.kmeans import KMeans
from gemseo.mlearning.transformers.scaler.min_max_scaler import MinMaxScaler

# Cluster locations
LOCS = array([[1.0, 0.0], [0.0, 1.0], [1.5, 1.5]])

# Cluster covariance matrices
SCALES = array([
    [[0.10, 0.00], [0.00, 0.05]],
    [[0.05, 0.01], [0.01, 0.10]],
    [[0.10, 0.05], [0.05, 0.10]],
])

# Number of samples in each cluster
N_SAMPLES = [50, 50, 50]

# Test value
VALUE = {"x_1": [0], "x_2": [0]}
ARRAY_VALUE = array([0, 0])

# Test values (centers of the clusters)
VALUES = {"x_1": LOCS[:, [0]], "x_2": LOCS[:, [1]]}


@pytest.fixture
def samples() -> tuple[ndarray, ndarray, list[int]]:
    """The description of the samples used to generate the learning dataset.

    It consists of three clusters from normal distributions.
    """
    # Check that the parameters conform
    assert len(SCALES) == len(LOCS)
    assert len(N_SAMPLES) == len(LOCS)
    for i in range(len(LOCS)):
        assert all(eigvals(SCALES[i]) > 0)  # Positive definite covariance
    return LOCS, SCALES, N_SAMPLES


@pytest.fixture
def dataset(samples) -> IODataset:
    """The dataset used to train the GaussianMixture.

    It consists of three clusters from normal distributions.
    """
    # Fix seed for consistency
    rng = default_rng(12345)

    # Unpack means, covariance matrices and number of samples
    locs, scales, n_samples = samples

    # Concatenate samples from the different normal distributions
    n_clusters = len(locs)
    data = array([[]])
    for i in range(n_clusters):
        temp = rng.multivariate_normal(locs[i], scales[i], n_samples[i])
        data = temp if i == 0 else vstack((data, temp))

    variables = ["x_1", "x_2"]

    sample = IODataset.from_array(data, variables)
    sample.name = "dataset_name"

    return sample


@pytest.fixture(params=["parameters", "x_1"])
def transformer_key(request):
    """The name of the group or variable to transform."""
    return request.param


@pytest.fixture(params=[False, True])
def fit_transformers(request):
    """Whether to fit the transformers during the training stage."""
    return request.param


@pytest.fixture
def model(dataset):
    """A trained KMeans with parameters scaling."""
    kmeans = KMeans(dataset, n_clusters=3)
    kmeans.learn()
    return kmeans


@pytest.fixture
def model_with_transform(dataset, transformer_key, fit_transformers):
    """A trained KMeans with parameters scaling."""
    kmeans = KMeans(
        dataset, transformer={transformer_key: MinMaxScaler()}, n_clusters=3
    )
    kmeans.learn()
    if not fit_transformers:
        kmeans = KMeans(dataset, transformer=kmeans.transformer, n_clusters=3)
        kmeans.learn(fit_transformers=False)
    return kmeans


def test_constructor(dataset) -> None:
    """Test construction."""
    algo = KMeans(dataset)
    assert algo.algo is not None
    assert algo.SHORT_ALGO_NAME == "KMeans"
    assert algo.LIBRARY == "scikit-learn"


def test_learn(dataset) -> None:
    """Test learn."""
    n_clusters = 5
    kmeans = KMeans(dataset, n_clusters=n_clusters)
    another_kmeans = KMeans(dataset, var_names=["x_1"], n_clusters=n_clusters)
    yet_another_kmeans = KMeans(dataset, var_names=["x_2"], n_clusters=n_clusters)
    kmeans.learn()
    another_kmeans.learn()
    yet_another_kmeans.learn(samples=[2, 4, 5, 1, 10, 11, 12])
    for km_model in [kmeans, another_kmeans, yet_another_kmeans]:
        assert km_model.algo is not None
        assert km_model.labels is not None
        assert km_model.n_clusters == n_clusters


def test_predict(model) -> None:
    """Test prediction."""
    prediction = model.predict(VALUE)
    predictions = model.predict(VALUES)
    assert isinstance(prediction, (int, integer))
    assert isinstance(predictions, ndarray)
    assert len(predictions.shape) == 1
    assert predictions[0] != predictions[1]
    assert predictions[0] != predictions[2]
    assert predictions[1] != predictions[2]


def test_predict_with_transform(model_with_transform) -> None:
    """Test prediction."""
    prediction = model_with_transform.predict(VALUE)
    predictions = model_with_transform.predict(VALUES)
    assert isinstance(prediction, (int, integer))
    assert isinstance(predictions, ndarray)
    assert len(predictions.shape) == 1
    assert predictions[0] != predictions[1]
    assert predictions[0] != predictions[2]
    assert predictions[1] != predictions[2]


@pytest.mark.parametrize("hard", [True, False])
def test_predict_proba(model, hard) -> None:
    """Test prediction."""
    proba = model.predict_proba(VALUE, hard)
    probas = model.predict_proba(VALUES, hard)
    assert (proba == model.predict_proba(ARRAY_VALUE, hard)).all()
    assert isinstance(proba, ndarray)
    assert isinstance(probas, ndarray)
    assert len(proba.shape) == 1
    assert len(probas.shape) == 2
    assert allclose(proba.sum(), 1)
    assert allclose(probas.sum(axis=1), 1)
    assert not allclose(probas[0], probas[1])
    assert not allclose(probas[0], probas[2])
    assert not allclose(probas[1], probas[2])
