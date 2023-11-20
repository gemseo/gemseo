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

import pytest
from numpy import array

from gemseo.datasets.io_dataset import IODataset
from gemseo.mlearning.core.ml_algo import MLAlgo
from gemseo.mlearning.quality_measures.cluster_measure import MLClusteringMeasure
from gemseo.mlearning.quality_measures.cluster_measure import (
    MLPredictiveClusteringMeasure,
)
from gemseo.mlearning.quality_measures.error_measure import MLErrorMeasure
from gemseo.mlearning.quality_measures.quality_measure import MLQualityMeasure
from gemseo.mlearning.quality_measures.quality_measure import MLQualityMeasureFactory
from gemseo.mlearning.quality_measures.r2_measure import R2Measure
from gemseo.utils.testing.helpers import concretize_classes


@pytest.fixture(scope="module")
def dataset() -> IODataset:
    """The learning dataset."""
    data = IODataset(dataset_name="the_dataset")
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


def test_is_better():
    class MLQualityMeasureToMinimize(MLQualityMeasure):
        SMALLER_IS_BETTER = True

    class MLQualityMeasureToMaximize(MLQualityMeasure):
        SMALLER_IS_BETTER = False

    assert MLQualityMeasureToMinimize.is_better(1, 2)
    assert MLQualityMeasureToMaximize.is_better(2, 1)


def test_factory():
    """Check that the factory of MLQualityMeasure works correctly."""
    assert "MSEMeasure" in MLQualityMeasureFactory().class_names


@pytest.mark.parametrize(
    ("cls", "old", "new"),
    [
        (MLQualityMeasure, "evaluate_loo", "compute_leave_one_out_measure"),
        (MLErrorMeasure, "evaluate_learn", "compute_learning_measure"),
        (MLErrorMeasure, "evaluate_test", "compute_test_measure"),
        (MLErrorMeasure, "evaluate_kfolds", "compute_cross_validation_measure"),
        (MLErrorMeasure, "evaluate_loo", "compute_leave_one_out_measure"),
        (MLErrorMeasure, "evaluate_bootstrap", "compute_bootstrap_measure"),
        (MLClusteringMeasure, "evaluate_learn", "compute_learning_measure"),
        (MLPredictiveClusteringMeasure, "evaluate_test", "compute_test_measure"),
        (
            MLPredictiveClusteringMeasure,
            "evaluate_kfolds",
            "compute_cross_validation_measure",
        ),
        (
            MLPredictiveClusteringMeasure,
            "evaluate_loo",
            "compute_leave_one_out_measure",
        ),
        (
            MLPredictiveClusteringMeasure,
            "evaluate_bootstrap",
            "compute_bootstrap_measure",
        ),
        (
            R2Measure,
            "evaluate_kfolds",
            "compute_cross_validation_measure",
        ),
        (
            R2Measure,
            "evaluate_bootstrap",
            "compute_bootstrap_measure",
        ),
    ],
)
def test_deprecated_evaluate_xxx(cls, old, new):
    """Check that the aliases of the deprecated evaluation methods are correct."""
    assert getattr(cls, old) == getattr(cls, new)
