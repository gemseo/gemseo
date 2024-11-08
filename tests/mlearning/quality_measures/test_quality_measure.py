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
from gemseo.mlearning.core.quality.base_ml_algo_quality import BaseMLAlgoQuality
from gemseo.mlearning.core.quality.factory import MLAlgoQualityFactory
from gemseo.utils.testing.helpers import concretize_classes

from ..core.test_ml_algo import DummyMLAlgo


@pytest.fixture(scope="module")
def dataset() -> IODataset:
    """The learning dataset."""
    data = IODataset(dataset_name="the_dataset")
    data.add_variable("x", array([[1]]))
    return data


@pytest.fixture(scope="module")
def measure(dataset) -> BaseMLAlgoQuality:
    """The quality measure related to a trained machine learning algorithm."""
    with concretize_classes(BaseMLAlgoQuality, DummyMLAlgo):
        return BaseMLAlgoQuality(DummyMLAlgo(dataset))


@pytest.mark.parametrize("fit_transformers", [False, True])
def test_constructor(fit_transformers, dataset) -> None:
    """Test construction."""
    with concretize_classes(BaseMLAlgoQuality, DummyMLAlgo):
        measure = BaseMLAlgoQuality(
            DummyMLAlgo(dataset), fit_transformers=fit_transformers
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
    assert MLAlgoQualityFactory().is_available("MSEMeasure")
