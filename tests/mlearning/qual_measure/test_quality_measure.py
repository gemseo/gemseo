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
from __future__ import annotations

from unittest.mock import Mock

import pytest
from gemseo.core.dataset import Dataset
from gemseo.mlearning.core.ml_algo import MLAlgo
from gemseo.mlearning.qual_measure.quality_measure import MLQualityMeasure
from gemseo.mlearning.qual_measure.quality_measure import MLQualityMeasureFactory
from gemseo.utils.pytest_conftest import concretize_classes
from numpy import array
from numpy import array_equal


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    """The learning dataset."""
    data = Dataset("the_dataset")
    data.add_variable("x", array([[1]]))
    return data


@pytest.fixture(scope="module")
def measure(dataset) -> MLQualityMeasure:
    """The quality measure related to a trained machine learning algorithm."""
    with concretize_classes(MLQualityMeasure, MLAlgo):
        return MLQualityMeasure(MLAlgo(dataset))


@pytest.mark.parametrize("fit_transformers", [False, True])
def test_constructor(fit_transformers, dataset):
    """Test construction."""
    with concretize_classes(MLQualityMeasure, MLAlgo):
        measure = MLQualityMeasure(MLAlgo(dataset), fit_transformers=fit_transformers)

    assert measure.algo.learning_set.name == "the_dataset"
    assert measure._fit_transformers is fit_transformers


def test_evaluate_unknown_method(measure):
    """Check that an error is raised when evaluating an unknown method."""
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
    algo.learning_samples_indices = [0, 1, 2, 3, 4]
    return algo


@pytest.mark.parametrize("samples", [None, [0, 2, 3]])
@pytest.mark.parametrize("n_folds", [2, 3])
@pytest.mark.parametrize("randomize", [False, True])
def test_randomize_cv(algo_with_three_samples, samples, n_folds, randomize):
    """Check that randomized cross-validation works correctly."""
    with concretize_classes(MLQualityMeasure):
        measure = MLQualityMeasure(algo_with_three_samples)

    folds, final_samples = measure._compute_folds(samples, n_folds, randomize, None)
    assert len(folds) == n_folds
    assert set.union(*(set(fold) for fold in folds)) == set(final_samples)
    assert sum(len(fold) == 0 for fold in folds) == 0

    if samples is None:
        assert set(final_samples) == {0, 1, 2, 3, 4}
    else:
        assert set(final_samples) == {0, 2, 3}

    replicates = []
    for _ in range(10):
        _, final_samples = measure._compute_folds(samples, n_folds, randomize, None)
        replicates.append(final_samples.tolist())

    replicates = array(replicates)

    all_replicates_are_identical = max(replicates.var(0)) == 0
    if randomize:
        assert not all_replicates_are_identical
    else:
        assert all_replicates_are_identical


@pytest.mark.parametrize("seed", [None, 1])
def test_cross_validation_seed(measure, seed):
    """Check that the seed is correctly used by cross-validation."""
    _, samples_1 = measure._compute_folds([0, 1, 2, 3, 4], 5, True, seed)
    _, samples_2 = measure._compute_folds([0, 1, 2, 3, 4], 5, True, seed)
    if seed is not None:
        assert array_equal(samples_1, samples_2)
    else:
        assert not array_equal(samples_1, samples_2)
