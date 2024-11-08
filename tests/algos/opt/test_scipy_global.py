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
from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.case import TestCase

import pytest
from numpy import array

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt.factory import OptimizationLibraryFactory
from gemseo.algos.opt.scipy_global.scipy_global import ScipyGlobalOpt
from gemseo.algos.opt.scipy_global.settings.differential_evolution import (
    DIFFERENTIAL_EVOLUTION_Settings,
)
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.core.mdo_functions.mdo_function import MDOFunction
from gemseo.problems.optimization.power_2 import Power2
from gemseo.problems.optimization.rosenbrock import Rosenbrock
from gemseo.utils.testing.opt_lib_test_base import OptLibraryTestBase

if TYPE_CHECKING:
    from gemseo.algos.database import Database


@pytest.mark.xfail(reason="With scipy 1.11+")
class TestScipyGlobalOpt(TestCase):
    """"""

    OPT_LIB_NAME = "ScipyGlobalOpt"

    @staticmethod
    def get_problem():
        return Rosenbrock()

    def test_init(self) -> None:
        factory = OptimizationLibraryFactory()
        if factory.is_available(self.OPT_LIB_NAME):
            factory.create("DUAL_ANNEALING")


@pytest.fixture(scope="module")
def pow2_database() -> Database:
    """The database resulting from the Power2 problem resolution."""
    problem = Power2()
    OptimizationLibraryFactory().execute(problem, algo_name="SHGO", max_iter=20)
    return problem.database


@pytest.mark.parametrize("name", ["pow2", "ineq1", "ineq2", "eq"])
def test_function_history_length(name, pow2_database) -> None:
    assert len(pow2_database.get_function_history(name)) == len(pow2_database)


def get_settings(algo_name):
    settings = {
        "max_iter": 3000,
        "seed": 1,
    }

    if algo_name == "DIFFERENTIAL_EVOLUTION":
        settings["normalize_design_space"] = False
        settings["popsize"] = 5
        settings["mutation"] = (0.6, 1)
    elif algo_name == "SHGO":
        settings["n"] = 100
        settings["sampling_method"] = "sobol"
        settings["iters"] = 1
        del settings["seed"]
    return settings


suite_tests = OptLibraryTestBase()
for test_method in suite_tests.generate_test("ScipyGlobalOpt", get_settings):
    setattr(TestScipyGlobalOpt, test_method.__name__, test_method)


def test_differential_evolution_parallel():
    """Test that the Differential Evolution algorithm works in parallel."""
    problem = Rosenbrock()
    result = OptimizationLibraryFactory().execute(
        problem,
        algo_name="DIFFERENTIAL_EVOLUTION",
        settings_model=DIFFERENTIAL_EVOLUTION_Settings(
            max_iter=5,
            workers=2,
            popsize=2,
        ),
    )
    assert result.f_opt


@pytest.fixture
def unconstrained_problem() -> OptimizationProblem:
    """An unconstrained optimization problem"""
    design_space = DesignSpace()
    design_space.add_variable(
        "x", size=1, lower_bound=array([-1.0]), upper_bound=array([1.0])
    )
    problem = OptimizationProblem(design_space)
    problem.objective = MDOFunction(name="f", func=lambda x: x**2)
    return problem


@pytest.mark.parametrize("algorithm_name", ScipyGlobalOpt.ALGORITHM_INFOS)
def test_max_iter(algorithm_name, unconstrained_problem):
    """Test that the maximum number of iteration is monitored by GEMSEO."""
    ScipyGlobalOpt(algorithm_name).execute(unconstrained_problem, max_iter=10)
