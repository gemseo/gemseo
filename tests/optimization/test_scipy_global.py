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
from unittest import mock
from unittest.case import TestCase

import pytest
from numpy import array
from numpy.testing import assert_equal

from gemseo.core.function.array_function import ArrayFunction
from gemseo.optimization.factory import optimization_library_factory
from gemseo.optimization.problem import OptimizationProblem
from gemseo.optimization.scipy_global.scipy_global import ScipyGlobalOpt
from gemseo.optimization.scipy_global.settings.differential_evolution import (
    DIFFERENTIAL_EVOLUTION_Settings,
)
from gemseo.optimization.scipy_global.settings.dual_annealing import (
    DUAL_ANNEALING_Settings,
)
from gemseo.optimization.scipy_global.settings.shgo import SHGO_Settings
from gemseo.optimization.scipy_local.settings.slsqp import SLSQP_Settings
from gemseo.problem.optimization.power_2 import Power2
from gemseo.problem.optimization.rosenbrock import Rosenbrock
from gemseo.space.design import DesignSpace
from gemseo.util.testing.helper import assert_exception
from gemseo.util.testing.opt_lib_test_base import OptLibraryTestBase

if TYPE_CHECKING:
    from gemseo.core.problem.database import Database


@pytest.mark.xfail(reason="With scipy 1.11+")
class TestScipyGlobalOpt(TestCase):
    """"""

    OPT_LIB_NAME = "ScipyGlobalOpt"

    @staticmethod
    def get_problem():
        return Rosenbrock()

    def test_init(self) -> None:
        factory = optimization_library_factory
        if factory.is_available(self.OPT_LIB_NAME):
            factory.create("DUAL_ANNEALING")


@pytest.fixture(scope="module")
def pow2_database() -> Database:
    """The database resulting from the Power2 problem resolution."""
    problem = Power2()
    optimization_library_factory.execute(problem, settings=SHGO_Settings(max_iter=20))
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


def test_listener_is_removed_after_run():
    """Check that the objective/constraint listener does not outlive the run.

    `_evaluate_objective_and_constraints` dereferences `_original_problem`,
    which `_reset()` sets back to `None` once the run is over;
    left registered on the database,
    it made the next store on this problem fail,
    e.g. a further run or a call to `evaluate_functions`.
    """
    problem = Power2()
    optimization_library_factory.execute(
        problem,
        settings=DIFFERENTIAL_EVOLUTION_Settings(max_iter=3, popsize=2, seed=1),
    )
    problem.evaluate_functions(input_value=problem.input_space.get_current_value())
    result = optimization_library_factory.execute(problem, settings=SLSQP_Settings())
    assert result.x_opt is not None


def test_listener_is_removed_when_run_raises_before_the_optimizer_call():
    """Check that the listener does not outlive a run raising before optimizing.

    The listener used to be registered outside the try/finally clause removing it,
    so an exception raised between the registration and the call to the SciPy
    optimizer, e.g. while building its settings, left it on the database.
    """
    problem = Power2()
    with (
        mock.patch.object(
            DesignSpace, "get_integer_mask", side_effect=RuntimeError("boom")
        ),
        pytest.raises(RuntimeError, match="boom"),
    ):
        optimization_library_factory.execute(
            problem, settings=DIFFERENTIAL_EVOLUTION_Settings(max_iter=3)
        )

    result = optimization_library_factory.execute(problem, settings=SLSQP_Settings())
    assert result.x_opt is not None


def test_differential_evolution_parallel():
    """Test that the Differential Evolution algorithm works in parallel."""
    problem = Rosenbrock()
    result = optimization_library_factory.execute(
        problem,
        settings=DIFFERENTIAL_EVOLUTION_Settings(
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
    design_space.add_variable("x", lower_bound=array([-1.0]), upper_bound=array([1.0]))
    problem = OptimizationProblem(design_space)
    problem.objective = ArrayFunction(lambda x: x**2, name="f")
    return problem


@pytest.mark.parametrize("algorithm_name", ScipyGlobalOpt.ALGORITHM_INFOS)
def test_max_iter(algorithm_name, unconstrained_problem):
    """Test that the maximum number of iteration is monitored by GEMSEO."""
    lib = ScipyGlobalOpt(algorithm_name)
    settings = lib.ALGORITHM_INFOS[algorithm_name].settings_class(max_iter=10)
    lib.execute(unconstrained_problem, settings=settings)


def _create_integer_problem() -> OptimizationProblem:
    """Create an optimization problem with an integer and a continuous variable.

    Returns:
        The problem.
    """
    design_space = DesignSpace()
    design_space.add_variable(
        "x", type_="integer", lower_bound=-5, upper_bound=5, value=0
    )
    design_space.add_variable("y", lower_bound=-5.0, upper_bound=5.0, value=0.0)
    problem = OptimizationProblem(design_space)
    problem.objective = ArrayFunction(
        lambda x: array([(x[0] - 2.0) ** 2 + (x[1] - 3.3) ** 2]),
        name="f",
    )
    return problem


@pytest.mark.parametrize("normalize_design_space", [False, True])
def test_differential_evolution_keeps_integer_variables_integral(
    normalize_design_space,
) -> None:
    """Check that DE keeps an integer variable integral at every evaluation.

    `DIFFERENTIAL_EVOLUTION` declares handling integer variables and is now
    told, through SciPy's own `integrality` option, which components are
    integer, so it explores that variable over the integers only, whatever
    `normalize_design_space` asks for, instead of continuously.
    """
    problem = _create_integer_problem()

    optimization_library_factory.execute(
        problem,
        settings=DIFFERENTIAL_EVOLUTION_Settings(
            max_iter=3,
            popsize=2,
            seed=1,
            normalize_design_space=normalize_design_space,
        ),
    )

    integer_components = array([
        x_vect[0] for x_vect in problem.database.get_x_vect_history()
    ])
    assert_equal(integer_components, integer_components.astype(int))
    x_opt = problem.solution.x_opt
    assert x_opt[0] == int(x_opt[0])


def test_dual_annealing_rejects_integer_variables_without_relaxation(snapshot) -> None:
    """Check that DUAL_ANNEALING raises on an integer variable unless relaxed.

    `DUAL_ANNEALING` does not handle integer variables,
    so it raises unless `relax_integer_variables` is `True`.
    """
    problem = _create_integer_problem()

    with assert_exception(ValueError, snapshot):
        optimization_library_factory.execute(
            problem, settings=DUAL_ANNEALING_Settings(max_iter=3)
        )


def test_dual_annealing_relaxes_integer_variables_when_asked_to() -> None:
    """Check that DUAL_ANNEALING runs on a relaxed integer variable.

    With `relax_integer_variables=True`, the integer variable is relaxed to a
    float one and explored continuously; the optimum the driver writes back
    into the design space is rounded, and `x_opt_projected` holds that
    rounded, integral, value.
    """
    problem = _create_integer_problem()

    optimization_library_factory.execute(
        problem,
        settings=DUAL_ANNEALING_Settings(max_iter=3, relax_integer_variables=True),
    )

    x_opt_projected = problem.solution.x_opt_projected
    assert x_opt_projected[0] == int(x_opt_projected[0])


@pytest.mark.parametrize("algorithm_name", ScipyGlobalOpt.ALGORITHM_INFOS)
def test_scipy_global_rejects_discrete_variables_without_relaxation(
    algorithm_name, snapshot
) -> None:
    """Check that no SciPy global algorithm handles a discrete variable.

    None of `DIFFERENTIAL_EVOLUTION`, `DUAL_ANNEALING` or `SHGO` declares
    handling discrete variables, so each of them raises on one unless
    `relax_discrete_variables` is `True`.
    """
    design_space = DesignSpace()
    design_space.add_discrete_variable("x", [1, 2, 3], value=1)
    problem = OptimizationProblem(design_space)
    problem.objective = ArrayFunction(lambda x: array([(x[0] - 2.0) ** 2]), name="f")

    lib = ScipyGlobalOpt(algorithm_name)
    settings = lib.ALGORITHM_INFOS[algorithm_name].settings_class(max_iter=3)
    with assert_exception(ValueError, snapshot):
        lib.execute(problem, settings=settings)
