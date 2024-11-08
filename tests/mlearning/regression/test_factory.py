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
"""Unit test for RegressorFactory class in gemseo.mlearning.regression.factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from gemseo.algos.design_space import DesignSpace
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.mlearning.regression.algos.factory import RegressorFactory
from gemseo.mlearning.regression.algos.linreg import LinearRegressor
from gemseo.scenarios.doe_scenario import DOEScenario

if TYPE_CHECKING:
    from gemseo.datasets.dataset import Dataset

LEARNING_SIZE = 9


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


def test_constructor() -> None:
    """Test factory constructor."""
    assert {
        "GaussianProcessRegressor",
        "LinearRegressor",
        "MOERegressor",
        "PCERegressor",
        "PolynomialRegressor",
        "RBFRegressor",
        "RandomForestRegressor",
    } <= set(RegressorFactory().class_names)


def test_create(dataset) -> None:
    """Test the creation of a model from data."""
    factory = RegressorFactory()
    assert isinstance(factory.create("LinearRegressor", data=dataset), LinearRegressor)


def test_is_available() -> None:
    """Test the existence of a regression model."""
    factory = RegressorFactory()
    assert factory.is_available("LinearRegressor")
    assert not factory.is_available("Dummy")
