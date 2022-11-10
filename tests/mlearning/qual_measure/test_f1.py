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
from gemseo.core.dataset import Dataset
from gemseo.mlearning.classification.knn import KNNClassifier
from gemseo.mlearning.core.ml_algo import MLAlgo
from gemseo.mlearning.qual_measure.f1_measure import F1Measure
from gemseo.utils.pytest_conftest import concretize_classes
from numpy import arange
from numpy import array


@pytest.fixture
def dataset() -> Dataset:
    """The dataset used to train the classification algorithms."""
    input_data = 1.0 * arange(63).reshape((21, 3))
    output_data = array([[0], [1], [2]]).repeat(7, axis=0)
    dataset_ = Dataset()
    dataset_.add_group(Dataset.INPUT_GROUP, input_data)
    dataset_.add_group(Dataset.OUTPUT_GROUP, output_data)
    return dataset_


@pytest.fixture
def dataset_test() -> Dataset:
    """The dataset used to test the performance classification algorithms."""
    input_data = 1.0 * arange(18).reshape((6, 3))
    output_data = array([[0], [1], [2]]).repeat(2, axis=0)
    dataset_ = Dataset()
    dataset_.add_group(Dataset.INPUT_GROUP, input_data)
    dataset_.add_group(Dataset.OUTPUT_GROUP, output_data)
    return dataset_


def test_constructor(dataset):
    """Test construction."""
    with concretize_classes(MLAlgo):
        algo = MLAlgo(dataset)

    measure = F1Measure(algo)
    assert measure.algo is not None
    assert measure.algo.learning_set is dataset


def test_evaluate_learn(dataset):
    """Test evaluate learn method."""
    algo = KNNClassifier(dataset)
    measure = F1Measure(algo)
    measure.evaluate("learn", multioutput=False)


def test_evaluate_test(dataset, dataset_test):
    """Test evaluate test method."""
    algo = KNNClassifier(dataset)
    measure = F1Measure(algo)
    measure.evaluate("test", test_data=dataset_test, multioutput=False)


def test_evaluate_loo(dataset):
    """Test evaluate leave one out method."""
    algo = KNNClassifier(dataset)
    measure = F1Measure(algo)
    measure.evaluate("loo", multioutput=False)


def test_evaluate_kfolds(dataset):
    """Test evaluate k-folds method."""
    algo = KNNClassifier(dataset)
    measure = F1Measure(algo)
    measure.evaluate("kfolds", multioutput=False)


def test_evaluate_bootstrap(dataset):
    """Test evaluate bootstrap method."""
    algo = KNNClassifier(dataset)
    measure = F1Measure(algo)
    measure.evaluate("bootstrap", multioutput=False)
