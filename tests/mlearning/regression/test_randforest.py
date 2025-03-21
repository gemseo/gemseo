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
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Test random forest regression module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from numpy import allclose
from numpy import array

from gemseo.algos.design_space import DesignSpace
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.mlearning.regression.algos.random_forest import RandomForestRegressor
from gemseo.scenarios.doe_scenario import DOEScenario

if TYPE_CHECKING:
    from gemseo.datasets.dataset import Dataset

LEARNING_SIZE = 9

INPUT_VALUE = {"x_1": array([1]), "x_2": array([2])}
INPUT_VALUES = {"x_1": array([[1], [0], [3]]), "x_2": array([[2], [1], [1]])}


@pytest.fixture
def dataset() -> Dataset:
    """The dataset used to train the regression algorithms."""
    discipline = AnalyticDiscipline({"y_1": "1+2*x_1+3*x_2", "y_2": "-1-2*x_1-3*x_2"})
    discipline.set_cache(discipline.CacheType.MEMORY_FULL)
    design_space = DesignSpace()
    design_space.add_variable("x_1", lower_bound=0.0, upper_bound=1.0)
    design_space.add_variable("x_2", lower_bound=0.0, upper_bound=1.0)
    scenario = DOEScenario(
        [discipline], "y_1", design_space, formulation_name="DisciplinaryOpt"
    )
    scenario.execute(algo_name="PYDOE_FULLFACT", n_samples=LEARNING_SIZE)
    return discipline.cache.to_dataset("dataset_name")


@pytest.fixture
def model(dataset) -> RandomForestRegressor:
    """A trained RandomForestRegressor."""
    random_forest = RandomForestRegressor(dataset)
    random_forest.learn()
    return random_forest


@pytest.fixture
def model_1d_output(dataset) -> RandomForestRegressor:
    """A trained RandomForestRegressor with only y_1 as outputs."""
    random_forest = RandomForestRegressor(dataset, output_names=["y_1"])
    random_forest.learn()
    return random_forest


def test_constructor(dataset) -> None:
    """Test construction."""
    model_ = RandomForestRegressor(dataset)
    assert model_.algo is not None


def test_learn(dataset) -> None:
    """Test learn."""
    model_ = RandomForestRegressor(dataset)
    model_.learn()
    assert model_.algo is not None
    assert model_.SHORT_ALGO_NAME == "RF"
    assert model_.LIBRARY == "scikit-learn"


def test_prediction(model) -> None:
    """Test prediction."""
    prediction = model.predict(INPUT_VALUE)
    predictions = model.predict(INPUT_VALUES)
    assert isinstance(prediction, dict)
    assert isinstance(predictions, dict)
    assert "y_1" in prediction
    assert "y_2" in prediction
    assert "y_1" in predictions
    assert "y_2" in predictions
    assert prediction["y_1"].shape == (1,)
    assert prediction["y_2"].shape == (1,)
    assert predictions["y_1"].shape == (3, 1)
    assert predictions["y_2"].shape == (3, 1)
    assert allclose(prediction["y_1"], -prediction["y_2"])
    assert allclose(predictions["y_1"], -predictions["y_2"])


def test_model_1d_output(model_1d_output) -> None:
    """Test the case where n_outputs=1, a particular case for random forest."""
    prediction = model_1d_output.predict(INPUT_VALUE)
    predictions = model_1d_output.predict(INPUT_VALUES)
    assert isinstance(prediction, dict)
    assert isinstance(predictions, dict)
    assert "y_1" in prediction
    assert "y_2" not in prediction
    assert "y_1" in predictions
    assert "y_2" not in predictions
    assert prediction["y_1"].shape == (1,)
    assert predictions["y_1"].shape == (3, 1)
