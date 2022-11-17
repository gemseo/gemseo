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
#
# Contributors:
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import logging
from sys import platform

import pytest
from gemseo.algos.database import Database
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.doe_factory import DOEFactory
from gemseo.algos.doe.doe_lib import DOELibrary
from gemseo.algos.doe.lib_openturns import OpenTURNS
from gemseo.algos.doe.lib_pydoe import PyDOE
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.api import create_discipline
from gemseo.api import create_scenario
from gemseo.api import execute_algo
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.problems.analytical.power_2 import Power2
from numpy import array

FACTORY = DOEFactory()


@pytest.fixture
def doe():
    pytest.mark.skipif(
        FACTORY.is_available("PyDOE"), reason="skipped because PyDOE is missing"
    )
    return FACTORY.create("PyDOE")


def test_fail_sample(doe):
    problem = Power2(exception_error=True)
    doe.execute(problem, "lhs", n_samples=4)


def test_evaluate_samples(doe):
    problem = Power2()
    doe.execute(problem, "fullfact", n_samples=2, wait_time_between_samples=1)


def test_evaluate_samples_multiproc(doe):
    problem = Power2()
    n_samples = 8
    doe.execute(
        problem,
        "fullfact",
        n_samples=n_samples,
        n_processes=2,
        wait_time_between_samples=0.01,
        eval_jac=True,
    )
    new_pb = Power2()
    x_history = problem.database.get_x_history()
    assert len(x_history) == n_samples
    for sample in x_history:
        val_ref = new_pb.objective(sample)
        val_sample = problem.database.get_f_of_x("pow2", sample)
        assert val_ref == val_sample

        grad_ref = new_pb.objective.jac(sample)
        grad_sample = problem.database.get_f_of_x("@pow2", sample)
        assert (grad_ref == grad_sample).all()


def compute_obj_and_obs(x: float = 0.0) -> tuple[float, float]:
    """Compute the objective and observable variables.

    Args:
         x: The input x value.

    Returns:
        obj: The objective value.
        obs: The observable value.
    """
    obj = x
    obs = x + 1.0
    return obj, obs


def test_evaluate_samples_multiproc_with_observables(doe):
    """Evaluate a DoE in // with multiprocessing and with observables."""

    disc = create_discipline("AutoPyDiscipline", py_func=compute_obj_and_obs)
    disc.cache = None
    design_space = DesignSpace()
    design_space.add_variable("x", l_b=0.0, u_b=3.0, value=2.0)

    scenario = create_scenario(
        [disc],
        design_space=design_space,
        objective_name="obj",
        formulation="DisciplinaryOpt",
        scenario_type="DOE",
    )

    samples = array(list([float(i)] for i in range(4)))
    scenario.add_observable("obs")
    scenario.execute(
        {"algo": "CustomDOE", "algo_options": {"n_processes": 2, "samples": samples}}
    )

    database = scenario.formulation.opt_problem.database
    for i, (x, data) in enumerate(database.items()):
        assert x.wrapped[0] == pytest.approx(float(i))
        assert data["obj"] == float(i)
        assert data["obs"] == float(i + 1)

    # In multi-processing mode,
    # the disciplinary calls are only made on the worker processes
    # Under Linux, the counters are updated from the subprocesses counters,
    # Under Windows, the discipline counters on the main process are not updated.
    # Without leveraging the cache mechanism,
    # the discipline shall be called 8 times.
    if platform == "win32":
        assert disc.n_calls == 0
    else:
        assert disc.n_calls == 8


def test_phip_criteria():
    """Check that the phi-p criterion is well implemented."""
    power = 3.0
    samples = array([[0.0, 0.0], [0.0, 2.0], [0.0, 3.0]])
    expected = sum(val ** (-power) for val in [2.0, 3.0, 1.0]) ** (1.0 / power)
    assert DOELibrary.compute_phip_criteria(samples, power) == expected


@pytest.fixture(scope="module")
def variables_space():
    """A mock design space."""
    design_space = DesignSpace()
    design_space.add_variable("x", l_b=0.0, u_b=2.0, value=1.0)
    design_space.add_variable("y", l_b=-1.0, u_b=1.0, value=0.0)
    return design_space


def test_compute_doe_transformed(variables_space):
    """Check the computation of a transformed DOE in a variables space."""
    doe = PyDOE()
    doe.algo_name = "fullfact"
    points = doe.compute_doe(variables_space, size=4, unit_sampling=True)
    assert (points == array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])).all()


def test_compute_doe_nontransformed(variables_space):
    """Check the computation of a non-transformed DOE in a variables space."""
    doe = PyDOE()
    doe.algo_name = "fullfact"
    points = doe.compute_doe(variables_space, size=4)
    assert (points == array([[0.0, -1.0], [2.0, -1.0], [0.0, 1.0], [2.0, 1.0]])).all()


@pytest.fixture(scope="module")
def doe_database(request) -> Database:
    """The DOE-based database with either deterministic or random variables."""
    if request.param:
        space = ParameterSpace()
        space.add_random_variable("var", "OTNormalDistribution")
    else:
        space = DesignSpace()
        space.add_variable("var", l_b=-3.0, u_b=4.0, value=1.0)

    problem = OptimizationProblem(space)
    problem.objective = MDOFunction(lambda x: x, "func")
    execute_algo(
        problem, "CustomDOE", samples=array([[-2.0], [0.0], [1.0]]), algo_type="doe"
    )
    return problem.database


@pytest.mark.parametrize("doe_database", [True, False], indirect=["doe_database"])
@pytest.mark.parametrize("var", [-2.0, 0.0, 1.0])
def test_transformation(doe_database, var):
    """Check that the transformation of variables works correctly.

    For the deterministic variables,
    the transformation is affine,
    based on the bounds of the variables.

    For the uncertain variables,
    the transformation is probabilistic,
    based on inverse transformation sampling.
    """
    assert doe_database[array([var])]["func"] == array([var])


def test_pre_run_debug(doe, caplog):
    """Check a DEBUG message logged just after sampling the input unit hypercube."""
    caplog.set_level("DEBUG")
    problem = Power2()
    doe.execute(problem, "lhs", n_samples=2)
    message = (
        "The DOE algorithm lhs of PyDOE has generated 2 samples "
        "in the input unit hypercube of dimension 3."
    )
    message_is_logged = False
    for (_, log_level, log_message) in caplog.record_tuples:
        if message in log_message:
            message_is_logged = True
            assert log_level == logging.DEBUG
            break

    assert message_is_logged


@pytest.mark.parametrize("algo_name", ["OT_MONTE_CARLO", "lhs"])
def test_seed(algo_name):
    """Check the use of the seed at the DOELibrary level."""
    problem = Power2()
    library = PyDOE() if algo_name == "lhs" else OpenTURNS()
    library.algo_name = algo_name

    # The DOELibrary has a seed and increments it at the beginning of each execution.
    assert library.seed == 0
    library.execute(
        problem,
        n_samples=2,
    )
    assert library.seed == 1
    assert len(problem.database) == 2

    # We execute a second time, still with the seed of the DOELibrary.
    # For that,
    # we need to reset the current iteration because max_iter is reached
    # (for DOELibrary, max_iter == n_samples).
    problem.reset(
        database=False, design_space=False, function_calls=False, preprocessing=False
    )
    library.execute(
        problem,
        n_samples=2,
    )
    assert library.seed == 2
    assert len(problem.database) == 4

    # We execute a third time,
    # with a seed passed as an option of the DOELibrary
    # and equal to the previous one.
    # By doing so,
    # the input samples will be the same and the functions won't be evaluated.
    problem.reset(
        database=False, design_space=False, function_calls=False, preprocessing=False
    )
    library.execute(problem, n_samples=2, seed=2)
    assert library.seed == 3
    # There is no new evaluation in the database:
    assert len(problem.database) == 4

    # Lastly, we check that the DOELibrary uses its own seed again.
    problem.reset(
        database=False, design_space=False, function_calls=False, preprocessing=False
    )
    library.execute(problem, n_samples=2)
    assert library.seed == 4
    # There are new evaluations in the database:
    assert len(problem.database) == 6
