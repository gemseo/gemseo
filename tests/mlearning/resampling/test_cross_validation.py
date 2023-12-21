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
"""Test for the module cross_validation."""

from __future__ import annotations

import pytest
from numpy import array
from numpy import array_equal
from numpy import linspace
from numpy import newaxis
from numpy.testing import assert_equal

from gemseo import SEED
from gemseo.datasets.io_dataset import IODataset
from gemseo.mlearning.regression.linreg import LinearRegressor
from gemseo.mlearning.resampling.cross_validation import CrossValidation
from gemseo.mlearning.resampling.split import Split
from gemseo.mlearning.resampling.splits import Splits


@pytest.fixture(scope="module")
def cross_validation(sample_indices) -> CrossValidation:
    """A cross-validation resampler."""
    return CrossValidation(sample_indices)


def test_default_properties(sample_indices):
    """Check the default values of the properties."""
    cross_validation = CrossValidation(sample_indices)
    assert_equal(cross_validation.sample_indices, sample_indices)
    assert cross_validation.seed == SEED
    assert cross_validation.n_folds == 5
    assert cross_validation.randomize is False
    assert array_equal(cross_validation.shuffled_sample_indices, sample_indices)
    assert_equal(
        cross_validation.splits,
        Splits(*[
            Split(array([2, 3, 4, 5, 6, 7, 8, 9]), array([0, 1])),
            Split(array([0, 1, 4, 5, 6, 7, 8, 9]), array([2, 3])),
            Split(array([0, 1, 2, 3, 6, 7, 8, 9]), array([4, 5])),
            Split(array([0, 1, 2, 3, 4, 5, 8, 9]), array([6, 7])),
            Split(array([0, 1, 2, 3, 4, 5, 6, 7]), array([8, 9])),
        ]),
    )


def test_properties_with_n_folds(sample_indices):
    """Check that the number and size of the folds depend on n_folds."""
    cross_validation = CrossValidation(sample_indices, n_folds=2)
    assert cross_validation.n_folds == 2
    assert_equal(
        cross_validation.splits,
        Splits(*[
            Split(array([5, 6, 7, 8, 9]), array([0, 1, 2, 3, 4])),
            Split(array([0, 1, 2, 3, 4]), array([5, 6, 7, 8, 9])),
        ]),
    )


def test_properties_with_randomize(sample_indices):
    """Check that the indices are shuffled before sampling when randomize is True."""
    cross_validation = CrossValidation(sample_indices, randomize=True)
    assert cross_validation.randomize is True
    assert not array_equal(cross_validation.shuffled_sample_indices, sample_indices)
    shuffled_sample_indices = cross_validation.shuffled_sample_indices.copy()
    shuffled_sample_indices.sort()
    assert array_equal(shuffled_sample_indices, sample_indices)
    assert not array_equal(next(iter(cross_validation.splits)).test, array([1, 2]))


def test_properties_with_none_seed(sample_indices):
    """Check that setting the seed at None makes the cross-validation random.

    i.e. two instances give different folds.
    """
    first_cross_validation = CrossValidation(sample_indices, randomize=True, seed=None)
    second_cross_validation = CrossValidation(sample_indices, randomize=True, seed=None)
    assert not array_equal(
        first_cross_validation.shuffled_sample_indices,
        second_cross_validation.shuffled_sample_indices,
    )


def test_properties_with_custom_seed(sample_indices):
    """Check that fixing the seed makes the cross-validation deterministic."""
    first_cross_validation = CrossValidation(sample_indices, randomize=True)
    second_cross_validation = CrossValidation(sample_indices, randomize=True, seed=2)
    third_cross_validation = CrossValidation(sample_indices, randomize=True, seed=2)
    assert not array_equal(
        first_cross_validation.shuffled_sample_indices,
        second_cross_validation.shuffled_sample_indices,
    )
    assert array_equal(
        second_cross_validation.shuffled_sample_indices,
        third_cross_validation.shuffled_sample_indices,
    )


@pytest.mark.parametrize("modify_learning", [False, True])
@pytest.mark.parametrize("modify_test", [False, True])
@pytest.mark.parametrize("add_fold", [False, True])
def test_eq(sample_indices, cross_validation, modify_learning, modify_test, add_fold):
    """Check that CrossValidation are equal if and only their folds are equal."""
    other_cross_validation = CrossValidation(sample_indices)
    iterator = iter(other_cross_validation.splits)
    first_split = next(iterator)
    if modify_learning:
        first_split.train[0] += 1

    if modify_test:
        first_split.test[0] += 1

    if add_fold:
        other_cross_validation._n_splits += 1
        splits = list(other_cross_validation.splits)
        splits.append(next(iterator))
        other_cross_validation._splits = Splits(*splits)

    assert (cross_validation != other_cross_validation) is (
        modify_learning or modify_test or add_fold
    )


@pytest.mark.parametrize("return_models", [False, True])
@pytest.mark.parametrize("predict", [False, True])
@pytest.mark.parametrize("stack_predictions", [False, True])
@pytest.mark.parametrize("fit_transformers", [False, True])
def test_execution(
    cross_validation, return_models, predict, stack_predictions, fit_transformers
):
    """Check that the method execute() works correctly."""
    dataset = IODataset()
    x = linspace(0, 1, 10)[:, newaxis]
    dataset.add_input_variable("x", x)
    dataset.add_output_variable("y", x)
    model = LinearRegressor(dataset)
    sub_models, predictions = cross_validation.execute(
        model,
        return_models,
        predict,
        stack_predictions,
        fit_transformers,
        False,
        model.input_data,
        model.output_data.shape,
    )
    assert (predictions == []) is not predict
    assert (sub_models == []) is not return_models

    if predict:
        if stack_predictions:
            assert predictions.shape == (10, 1)
        else:
            assert isinstance(predictions, list)
            assert len(predictions) == 5
            for prediction in predictions:
                assert prediction.shape == (2, 1)

    if return_models:
        assert isinstance(sub_models, list)
        assert len(sub_models) == 5
        assert len({id(sub_model) for sub_model in sub_models}) == 5


@pytest.mark.parametrize(
    ("n_folds", "name"), [(3, "CrossValidation"), (10, "LeaveOneOut")]
)
def test_name(sample_indices, n_folds, name):
    """Check the name of the CrossValidation instance."""
    assert CrossValidation(sample_indices, n_folds).name == name
