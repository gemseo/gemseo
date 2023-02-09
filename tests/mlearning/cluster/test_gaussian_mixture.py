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
"""Test Gaussian Mixture clustering model."""
from __future__ import annotations

import pytest
from gemseo.core.dataset import Dataset
from gemseo.mlearning.api import import_clustering_model
from gemseo.mlearning.cluster.gaussian_mixture import GaussianMixture
from gemseo.mlearning.transform.scaler.min_max_scaler import MinMaxScaler
from numpy import allclose
from numpy import array
from numpy import integer
from numpy import ndarray
from numpy import vstack
from numpy.linalg import eigvals
from numpy.random import multivariate_normal
from numpy.random import seed

# Cluster locations
LOCS = array([[1.0, 0.0], [0.0, 1.0], [1.5, 1.5]])

# Cluster covariance matrices
SCALES = array(
    [
        [[0.10, 0.00], [0.00, 0.05]],
        [[0.05, 0.01], [0.01, 0.10]],
        [[0.10, 0.05], [0.05, 0.10]],
    ]
)

# Number of samples in each cluster
N_SAMPLES = [50, 50, 50]

# Test value
VALUE = {"x_1": [0], "x_2": [0]}

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
def dataset(samples) -> Dataset:
    """The dataset used to train the GaussianMixture.

    It consists of three clusters from normal distributions.
    """
    # Fix seed for consistency
    seed(12345)

    # Unpack means, covariance matrices and number of samples
    locs, scales, n_samples = samples

    # Concatenate samples from the different normal distributions
    n_clusters = len(locs)
    data = array([[]])
    for i in range(n_clusters):
        temp = multivariate_normal(locs[i], scales[i], n_samples[i])
        if i == 0:
            data = temp
        else:
            data = vstack((data, temp))

    variables = ["x_1", "x_2"]

    sample = Dataset("dataset_name")
    sample.set_from_array(data, variables)
    return sample


@pytest.fixture
def model(dataset) -> GaussianMixture:
    """A trained GaussianMixture."""
    n_components = 3
    gaussian_mixture = GaussianMixture(dataset, n_components=n_components)
    gaussian_mixture.learn()
    return gaussian_mixture


@pytest.fixture
def model_with_transform(dataset) -> GaussianMixture:
    """A trained GaussianMixture with parameters scaling."""
    n_components = 3
    transformer = {"parameters": MinMaxScaler()}
    gaussian_mixture = GaussianMixture(
        dataset, transformer=transformer, n_components=n_components
    )
    gaussian_mixture.learn()
    return gaussian_mixture


def test_constructor(dataset):
    """Test construction."""
    gaussian_mixture = GaussianMixture(dataset)
    assert gaussian_mixture.algo is not None
    assert gaussian_mixture.SHORT_ALGO_NAME == "GMM"
    assert gaussian_mixture.LIBRARY == "scikit-learn"


def test_learn(dataset):
    """Test learn."""
    n_components = 5
    gaussian_mixture = GaussianMixture(dataset, n_components=n_components)
    another_gaussian_mixture = GaussianMixture(
        dataset, var_names=["x_1"], n_components=n_components
    )
    yet_another_gaussian_mixture = GaussianMixture(
        dataset, var_names=["x_2"], n_components=n_components
    )
    gaussian_mixture.learn()
    another_gaussian_mixture.learn()
    yet_another_gaussian_mixture.learn(samples=[2, 4, 5, 1, 10, 11, 12])
    for gm_model in [
        gaussian_mixture,
        another_gaussian_mixture,
        yet_another_gaussian_mixture,
    ]:
        assert gm_model.algo is not None
        assert gm_model.labels is not None
        assert gm_model.n_clusters == n_components


def test_predict(model):
    """Test prediction."""
    prediction = model.predict(VALUE)
    predictions = model.predict(VALUES)
    assert isinstance(prediction, (int, integer))
    assert isinstance(predictions, ndarray)
    assert len(predictions.shape) == 1
    assert predictions[0] != predictions[1]
    assert predictions[0] != predictions[2]
    assert predictions[1] != predictions[2]


def test_predict_with_transform(model_with_transform):
    """Test prediction."""
    prediction = model_with_transform.predict(VALUE)
    predictions = model_with_transform.predict(VALUES)
    assert isinstance(prediction, (int, integer))
    assert isinstance(predictions, ndarray)
    assert len(predictions.shape) == 1
    assert predictions[0] != predictions[1]
    assert predictions[0] != predictions[2]
    assert predictions[1] != predictions[2]


def test_predict_proba(model):
    """Test prediction."""
    for is_hard in [True, False]:
        proba = model.predict_proba(VALUE, is_hard)
        probas = model.predict_proba(VALUES, is_hard)
        assert isinstance(proba, ndarray)
        assert isinstance(probas, ndarray)
        assert len(proba.shape) == 1
        assert len(probas.shape) == 2
        assert allclose(proba.sum(), 1)
        assert allclose(probas.sum(axis=1), 1)
        assert not allclose(probas[0], probas[1])
        assert not allclose(probas[0], probas[2])
        assert not allclose(probas[1], probas[2])


def test_save_and_load(model, tmp_wd):
    """Test save and load."""
    dirname = model.save()
    imported_model = import_clustering_model(dirname)
    out1 = model.predict(VALUE)
    out2 = imported_model.predict(VALUE)
    assert out1 == out2
