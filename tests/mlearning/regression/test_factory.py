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
"""Unit test for RegressionModelFactory class in gemseo.mlearning.regression.factory."""
from __future__ import annotations

import pytest
from gemseo.algos.design_space import DesignSpace
from gemseo.core.dataset import Dataset
from gemseo.core.doe_scenario import DOEScenario
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.mlearning.regression.factory import RegressionModelFactory

LEARNING_SIZE = 9


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


def test_constructor():
    """Test factory constructor."""
    assert {
        "GaussianProcessRegressor",
        "LinearRegressor",
        "MOERegressor",
        "PCERegressor",
        "PolynomialRegressor",
        "RBFRegressor",
        "RandomForestRegressor",
    } <= set(RegressionModelFactory().models)


def test_create(dataset):
    """Test the creation of a model from data."""
    factory = RegressionModelFactory()
    linreg = factory.create("LinearRegressor", data=dataset)
    assert hasattr(linreg, "parameters")


def test_load(dataset, tmp_wd):
    """Test the loading of a model from data."""
    factory = RegressionModelFactory()
    linreg = factory.create("LinearRegressor", data=dataset)
    linreg.learn()
    dirname = linreg.save()
    loaded_linreg = factory.load(dirname)
    assert hasattr(loaded_linreg, "parameters")


def test_available_models():
    """Test the getter of available regression models."""
    factory = RegressionModelFactory()
    assert "LinearRegressor" in factory.models


def test_is_available():
    """Test the existence of a regression model."""
    factory = RegressionModelFactory()
    assert factory.is_available("LinearRegressor")
    assert not factory.is_available("Dummy")
