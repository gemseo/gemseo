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
"""Test quality measure module."""
from __future__ import division, unicode_literals

from unittest.mock import Mock

import pytest
from numpy import array

from gemseo.core.dataset import Dataset
from gemseo.mlearning.core.ml_algo import MLAlgo
from gemseo.mlearning.qual_measure.quality_measure import (
    MLQualityMeasure,
    MLQualityMeasureFactory,
)


@pytest.fixture(scope="module")
def measure():  # type: (...) -> MLQualityMeasure
    """The quality measure related to an trained machine learning algorithm."""
    dataset = Dataset("the_dataset")
    dataset.add_variable("x", array([[1]]))
    return MLQualityMeasure(MLAlgo(dataset))


def test_constructor(measure):
    """Test construction."""
    assert measure.algo is not None
    assert measure.algo.learning_set.name == "the_dataset"


def test_evaluate(measure):
    """Test evaluation of quality measure."""
    test_dataset = Dataset()
    with pytest.raises(NotImplementedError):
        measure.evaluate()
    with pytest.raises(NotImplementedError):
        measure.evaluate(MLQualityMeasure.LEARN)
    with pytest.raises(NotImplementedError):
        measure.evaluate(MLQualityMeasure.TEST, test_data=test_dataset)
    with pytest.raises(NotImplementedError):
        measure.evaluate(MLQualityMeasure.LOO)
    with pytest.raises(NotImplementedError):
        measure.evaluate(MLQualityMeasure.KFOLDS, n_folds=5)
    with pytest.raises(NotImplementedError):
        measure.evaluate(MLQualityMeasure.BOOTSTRAP, n_replicates=100)

    with pytest.raises(ValueError, match="The method 'foo' is not available"):
        measure.evaluate("foo")


def test_is_better():
    class MLQualityMeasureToMinimize(MLQualityMeasure):
        SMALLER_IS_BETTER = True

    class MLQualityMeasureToMaximize(MLQualityMeasure):
        SMALLER_IS_BETTER = False

    assert MLQualityMeasureToMinimize.is_better(1, 2)
    assert MLQualityMeasureToMaximize.is_better(2, 1)


def test_assure_samples(measure):
    assert measure._assure_samples([1, 2]).tolist() == [1, 2]
    assert measure._assure_samples(None) == array([0])


def test_factory():
    """Check that the factory of MLQualityMeasure works correctly."""
    assert "MSEMeasure" in MLQualityMeasureFactory().classes


@pytest.fixture
def algo_with_three_samples():
    learning_set = Mock()
    learning_set.n_samples = 5
    algo = Mock()
    algo.learning_set = learning_set
    return algo


@pytest.mark.parametrize("samples", [None, [0, 2, 3]])
@pytest.mark.parametrize("n_folds", [2, 3])
@pytest.mark.parametrize("randomize", [False, True])
def test_randomize_cv(algo_with_three_samples, samples, n_folds, randomize):
    """Check that randomized cross-validation works correctly."""
    measure = MLQualityMeasure(algo_with_three_samples)
    folds, final_samples = measure._compute_folds(samples, n_folds, randomize)
    assert len(folds) == n_folds
    assert set.union(*(set(fold) for fold in folds)) == set(final_samples)
    assert sum([len(fold) == 0 for fold in folds]) == 0

    if samples is None:
        assert set(final_samples) == {0, 1, 2, 3, 4}
    else:
        assert set(final_samples) == {0, 2, 3}

    replicates = []
    for _ in range(10):
        _, final_samples = measure._compute_folds(samples, n_folds, randomize)
        replicates.append(final_samples.tolist())

    replicates = array(replicates)

    all_replicates_are_identical = max(replicates.var(0)) == 0
    if randomize:
        assert not all_replicates_are_identical
    else:
        assert all_replicates_are_identical
