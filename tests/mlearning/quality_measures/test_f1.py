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
"""Test F1 measure."""

from __future__ import annotations

import pytest
from numpy import arange
from numpy import array

from gemseo.datasets.io_dataset import IODataset
from gemseo.mlearning.classification.knn import KNNClassifier
from gemseo.mlearning.core.ml_algo import MLAlgo
from gemseo.mlearning.quality_measures.f1_measure import F1Measure
from gemseo.utils.testing.helpers import concretize_classes


@pytest.fixture()
def dataset() -> IODataset:
    """The dataset used to train the classification algorithms."""
    input_data = 1.0 * arange(63).reshape((21, 3))
    output_data = array([[0], [1], [2]]).repeat(7, axis=0)
    dataset_ = IODataset()
    dataset_.add_group(dataset_.INPUT_GROUP, input_data)
    dataset_.add_group(dataset_.OUTPUT_GROUP, output_data)
    return dataset_


@pytest.fixture()
def dataset_test() -> IODataset:
    """The dataset used to test the performance classification algorithms."""
    input_data = 1.0 * arange(18).reshape((6, 3))
    output_data = array([[0], [1], [2]]).repeat(2, axis=0)
    dataset_ = IODataset()
    dataset_.add_group(dataset_.INPUT_GROUP, input_data)
    dataset_.add_group(dataset_.OUTPUT_GROUP, output_data)
    return dataset_


def test_constructor(dataset):
    """Test construction."""
    with concretize_classes(MLAlgo):
        algo = MLAlgo(dataset)

    measure = F1Measure(algo)
    assert measure.algo is not None
    assert measure.algo.learning_set is dataset


def test_compute_learning_measure(dataset):
    """Test evaluate learn method."""
    algo = KNNClassifier(dataset)
    measure = F1Measure(algo)
    measure.compute_learning_measure(multioutput=False)


def test_compute_test_measure(dataset, dataset_test):
    """Test evaluate test method."""
    algo = KNNClassifier(dataset)
    measure = F1Measure(algo)
    measure.compute_test_measure(test_data=dataset_test, multioutput=False)


def test_compute_leave_one_out_measure(dataset):
    """Test evaluate leave one out method."""
    algo = KNNClassifier(dataset)
    measure = F1Measure(algo)
    measure.compute_leave_one_out_measure(multioutput=False)


def test_compute_cross_validation_measure(dataset):
    """Test evaluate k-folds method."""
    algo = KNNClassifier(dataset)
    measure = F1Measure(algo)
    measure.compute_cross_validation_measure(multioutput=False)


def test_compute_bootstrap_measure(dataset):
    """Test evaluate bootstrap method."""
    algo = KNNClassifier(dataset)
    measure = F1Measure(algo)
    measure.compute_bootstrap_measure(multioutput=False)
