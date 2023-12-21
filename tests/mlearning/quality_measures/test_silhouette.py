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

from __future__ import annotations

import pytest
from numpy import arange

from gemseo.datasets.dataset import Dataset
from gemseo.mlearning.clustering.kmeans import KMeans
from gemseo.mlearning.quality_measures.silhouette_measure import SilhouetteMeasure


@pytest.fixture()
def dataset() -> Dataset:
    """The dataset used to train the regression algorithms."""
    data = arange(60).reshape((20, 3))
    dataset_ = Dataset()
    dataset_.add_variable("x", data, Dataset.PARAMETER_GROUP)
    return dataset_


@pytest.fixture()
def dataset_test() -> Dataset:
    """The dataset used to test the performance of the clustering algorithms."""
    data = arange(30).reshape((10, 3))
    dataset_ = Dataset()
    dataset_.add_variable("x", data, Dataset.PARAMETER_GROUP)
    return dataset_


@pytest.fixture()
def measure(dataset) -> SilhouetteMeasure:
    """A silhouette measure."""
    algo = KMeans(dataset, n_clusters=3)
    return SilhouetteMeasure(algo)


def test_constructor(measure, dataset):
    """Test construction."""
    assert measure.algo is not None
    assert measure.algo.learning_set is dataset


def test_compute_learning_measure(measure):
    """Test evaluate learn method."""
    quality = measure.compute_learning_measure(multioutput=False)
    assert quality > 0


def test_compute_learning_measure_fail(measure):
    """Test evaluate learn method; should fail if multioutput is True."""
    with pytest.raises(
        NotImplementedError,
        match="The SilhouetteMeasure does not support the multioutput case.",
    ):
        measure.compute_learning_measure(multioutput=True)


def test_compute_test_measure(measure, dataset_test):
    """Test evaluate test method."""
    with pytest.raises(NotImplementedError):
        measure.compute_test_measure(dataset_test, multioutput=False)


def test_compute_leave_one_out_measure(measure):
    """Test evaluate leave one out method."""
    with pytest.raises(NotImplementedError):
        measure.compute_leave_one_out_measure(multioutput=False)


def test_compute_cross_validation_measure(measure):
    """Test evaluate k-folds method."""
    with pytest.raises(NotImplementedError):
        measure.compute_cross_validation_measure(multioutput=False)


def test_compute_bootstrap_measure(measure):
    """Test evaluate bootstrap method."""
    with pytest.raises(NotImplementedError):
        measure.compute_bootstrap_measure(multioutput=False)
