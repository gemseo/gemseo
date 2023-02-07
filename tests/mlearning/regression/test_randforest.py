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

import pytest
from gemseo.algos.design_space import DesignSpace
from gemseo.core.dataset import Dataset
from gemseo.core.doe_scenario import DOEScenario
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.mlearning.api import import_regression_model
from gemseo.mlearning.regression.random_forest import RandomForestRegressor
from numpy import allclose
from numpy import array

LEARNING_SIZE = 9

INPUT_VALUE = {"x_1": array([1]), "x_2": array([2])}
INPUT_VALUES = {"x_1": array([[1], [0], [3]]), "x_2": array([[2], [1], [1]])}


@pytest.fixture
def dataset() -> Dataset:
    """The dataset used to train the regression algorithms."""
    discipline = AnalyticDiscipline({"y_1": "1+2*x_1+3*x_2", "y_2": "-1-2*x_1-3*x_2"})
    discipline.set_cache_policy(discipline.MEMORY_FULL_CACHE)
    design_space = DesignSpace()
    design_space.add_variable("x_1", l_b=0.0, u_b=1.0)
    design_space.add_variable("x_2", l_b=0.0, u_b=1.0)
    scenario = DOEScenario([discipline], "DisciplinaryOpt", "y_1", design_space)
    scenario.execute({"algo": "fullfact", "n_samples": LEARNING_SIZE})
    return discipline.cache.export_to_dataset("dataset_name")


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


def test_constructor(dataset):
    """Test construction."""
    model_ = RandomForestRegressor(dataset)
    assert model_.algo is not None


def test_learn(dataset):
    """Test learn."""
    model_ = RandomForestRegressor(dataset)
    model_.learn()
    assert model_.algo is not None
    assert model_.SHORT_ALGO_NAME == "RF"
    assert model_.LIBRARY == "scikit-learn"


def test_prediction(model):
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


def test_model_1d_output(model_1d_output):
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


def test_save_and_load(model, tmp_wd):
    """Test save and load."""
    dirname = model.save()
    imported_model = import_regression_model(dirname)
    out1 = model.predict(INPUT_VALUE)
    out2 = imported_model.predict(INPUT_VALUE)
    for name, value in out1.items():
        assert allclose(value, out2[name], 1e-3)
