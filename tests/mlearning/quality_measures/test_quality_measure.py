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
from gemseo.mlearning.clustering.quality.base_clusterer_quality import (
    BaseClustererQuality,
)
from gemseo.mlearning.clustering.quality.base_predictive_clusterer_quality import (
    BasePredictiveClustererQuality,
)
from gemseo.mlearning.core.algos.ml_algo import BaseMLAlgo
from gemseo.mlearning.core.quality.base_ml_algo_quality_ import BaseMLAlgoQuality
from gemseo.mlearning.core.quality.factory import MLAlgoQualityFactory
from gemseo.mlearning.regression.quality.base_regressor_quality import (
    BaseRegressorQuality,
)
from gemseo.mlearning.regression.quality.r2_measure import R2Measure
from gemseo.utils.testing.helpers import concretize_classes


@pytest.fixture(scope="module")
def dataset() -> IODataset:
    """The learning dataset."""
    data = IODataset(dataset_name="the_dataset")
    data.add_variable("x", array([[1]]))
    return data


@pytest.fixture(scope="module")
def measure(dataset) -> BaseMLAlgoQuality:
    """The quality measure related to a trained machine learning algorithm."""
    with concretize_classes(BaseMLAlgoQuality, BaseMLAlgo):
        return BaseMLAlgoQuality(BaseMLAlgo(dataset))


@pytest.mark.parametrize("fit_transformers", [False, True])
def test_constructor(fit_transformers, dataset) -> None:
    """Test construction."""
    with concretize_classes(BaseMLAlgoQuality, BaseMLAlgo):
        measure = BaseMLAlgoQuality(
            BaseMLAlgo(dataset), fit_transformers=fit_transformers
        )

    assert measure.algo.learning_set.name == "the_dataset"
    assert measure._fit_transformers is fit_transformers


def test_is_better() -> None:
    class MLQualityMeasureToMinimize(BaseMLAlgoQuality):
        SMALLER_IS_BETTER = True

    class MLQualityMeasureToMaximize(BaseMLAlgoQuality):
        SMALLER_IS_BETTER = False

    assert MLQualityMeasureToMinimize.is_better(1, 2)
    assert MLQualityMeasureToMaximize.is_better(2, 1)


def test_factory() -> None:
    """Check that the factory of BaseMLAlgoQuality works correctly."""
    assert "MSEMeasure" in MLAlgoQualityFactory().class_names


@pytest.mark.parametrize(
    ("cls", "old", "new"),
    [
        (BaseMLAlgoQuality, "evaluate_loo", "compute_leave_one_out_measure"),
        (BaseRegressorQuality, "evaluate_learn", "compute_learning_measure"),
        (BaseRegressorQuality, "evaluate_test", "compute_test_measure"),
        (BaseRegressorQuality, "evaluate_kfolds", "compute_cross_validation_measure"),
        (BaseRegressorQuality, "evaluate_loo", "compute_leave_one_out_measure"),
        (BaseRegressorQuality, "evaluate_bootstrap", "compute_bootstrap_measure"),
        (BaseClustererQuality, "evaluate_learn", "compute_learning_measure"),
        (BasePredictiveClustererQuality, "evaluate_test", "compute_test_measure"),
        (
            BasePredictiveClustererQuality,
            "evaluate_kfolds",
            "compute_cross_validation_measure",
        ),
        (
            BasePredictiveClustererQuality,
            "evaluate_loo",
            "compute_leave_one_out_measure",
        ),
        (
            BasePredictiveClustererQuality,
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
def test_deprecated_evaluate_xxx(cls, old, new) -> None:
    """Check that the aliases of the deprecated evaluation methods are correct."""
    assert getattr(cls, old) == getattr(cls, new)
