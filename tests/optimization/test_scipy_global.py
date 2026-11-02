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

import logging
import threading
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pickle import dumps
from pickle import loads
from typing import TYPE_CHECKING
from unittest import mock
from unittest.case import TestCase

import pytest
from numpy import array
from numpy import log
from numpy.testing import assert_equal
from pydantic import ValidationError
from scipy.optimize import OptimizeResult
from scipy.optimize import differential_evolution

from gemseo.core.function.array_function import ArrayFunction
from gemseo.core.problem.evaluation import EvaluationProblem
from gemseo.optimization.factory import optimization_library_factory
from gemseo.optimization.problem import OptimizationProblem
from gemseo.optimization.scipy_global import scipy_global
from gemseo.optimization.scipy_global.scipy_global import ScipyGlobalOpt
from gemseo.optimization.scipy_global.settings.differential_evolution import (
    DIFFERENTIAL_EVOLUTION_Settings,
)
from gemseo.optimization.scipy_global.settings.differential_evolution import (
    DifferentialEvolutionUpdating,
)
from gemseo.optimization.scipy_global.settings.dual_annealing import (
    DUAL_ANNEALING_Settings,
)
from gemseo.optimization.scipy_global.settings.shgo import SHGO_Settings
from gemseo.optimization.scipy_local.settings.slsqp import SLSQP_Settings
from gemseo.problem.optimization.power_2 import Power2
from gemseo.problem.optimization.rosenbrock import Rosenbrock
from gemseo.space.design import DesignSpace
from gemseo.util.multiprocessing import process_pool
from gemseo.util.multiprocessing import start_method
from gemseo.util.multiprocessing.start_method import MultiProcessingStartMethod
from gemseo.util.testing.helper import assert_exception
from gemseo.util.testing.opt_lib_test_base import OptLibraryTestBase

if TYPE_CHECKING:
    from gemseo.core.problem.database import Database


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


@pytest.mark.parametrize(
    ("in_calling_process", "enable_working_database"),
    [(False, False), (True, False), (True, True)],
    ids=[
        "worker_processes",
        "calling_process",
        "calling_process_with_working_database",
    ],
)
@pytest.mark.parametrize(
    "problem_class", [Rosenbrock, Power2], ids=["Rosenbrock", "Power2"]
)
@pytest.mark.parametrize(
    "settings_class",
    [
        partial(
            DIFFERENTIAL_EVOLUTION_Settings, popsize=2, seed=1, updating="deferred"
        ),
        partial(SHGO_Settings, n=8, iters=1),
    ],
    ids=["DIFFERENTIAL_EVOLUTION", "SHGO"],
)
def test_parallel(
    request,
    monkeypatch,
    problem_class,
    settings_class,
    in_calling_process,
    enable_working_database,
):
    """Check that the evaluations of the worker processes reach the database.

    The workers used to evaluate the functions on copies of the problem,
    whose databases were lost,
    so the starting point was returned as the optimum.
    The run is compared to a serial one;
    `max_iter` stops it in the middle of a batch of evaluations sent to the workers.
    The workers evaluate the objective of the unconstrained problem
    and the constraints of the constrained one with SHGO;
    DIFFERENTIAL_EVOLUTION does not use them with constraints.
    The evaluations are also made in the calling process,
    where coverage measures the worker functions,
    with and without a working database.
    """
    if in_calling_process:
        executors = request.getfixturevalue("serial_executors")

    if enable_working_database:
        monkeypatch.setattr(EvaluationProblem, "enable_working_database", True)

    max_iter = 10
    serial_problem = problem_class()
    optimization_library_factory.execute(
        serial_problem, settings=settings_class(max_iter=max_iter)
    )
    problem = problem_class()
    optimization_library_factory.execute(
        problem, settings=settings_class(max_iter=max_iter, workers=2)
    )
    uses_workers = problem_class is Rosenbrock or settings_class.func is SHGO_Settings
    database = problem.database
    serial_database = serial_problem.database
    assert len(database) == len(serial_database) == max_iter
    assert_equal(database.get_x_vect_history(), serial_database.get_x_vect_history())
    for name in serial_database.get_function_names():
        assert_equal(
            database.get_function_history(name),
            serial_database.get_function_history(name),
        )

    assert problem.evaluation_counter.current == max_iter
    assert_equal(problem.solution.x_opt, serial_problem.solution.x_opt)
    if in_calling_process:
        assert bool(executors) == uses_workers
        if uses_workers:
            assert executors[0].mapped_inputs


@pytest.mark.parametrize(
    ("workers", "updating"),
    [
        (1, DifferentialEvolutionUpdating.IMMEDIATE),
        (2, DifferentialEvolutionUpdating.DEFERRED),
    ],
)
def test_differential_evolution_default_updating(workers, updating):
    """Check the updating strategy passed to SciPy by default."""
    scipy_function = mock.Mock(wraps=differential_evolution)
    with mock.patch.object(
        ScipyGlobalOpt,
        "_ScipyGlobalOpt__names_to_functions",
        {"DIFFERENTIAL_EVOLUTION": scipy_function},
    ):
        optimization_library_factory.execute(
            Rosenbrock(),
            settings=DIFFERENTIAL_EVOLUTION_Settings(
                max_iter=2, popsize=2, workers=workers
            ),
        )

    assert scipy_function.call_args.kwargs["updating"] == updating


def test_differential_evolution_updating_from_string():
    """Check that the updating strategy can be set from its value."""
    settings = DIFFERENTIAL_EVOLUTION_Settings(updating="deferred")
    assert settings.updating is DifferentialEvolutionUpdating.DEFERRED


def _set_updating_after_workers() -> None:
    """Set the updating strategy of settings built with workers."""
    settings = DIFFERENTIAL_EVOLUTION_Settings(workers=2)
    settings.updating = "immediate"


def _set_workers_after_updating() -> None:
    """Set the workers of settings built with the immediate updating strategy."""
    settings = DIFFERENTIAL_EVOLUTION_Settings(updating="immediate")
    settings.workers = 2


def _build_with_updating_and_workers() -> None:
    """Build settings with the immediate updating strategy and workers."""
    DIFFERENTIAL_EVOLUTION_Settings(updating="immediate", workers=2)


def _build_with_updating_and_all_cpus() -> None:
    """Build settings with the immediate updating strategy and all the CPUs."""
    DIFFERENTIAL_EVOLUTION_Settings(updating="immediate", workers=-1)


@pytest.mark.parametrize(
    "function",
    [
        _set_updating_after_workers,
        _set_workers_after_updating,
        _build_with_updating_and_workers,
        _build_with_updating_and_all_cpus,
    ],
)
def test_differential_evolution_immediate_updating_with_workers(function, snapshot):
    """Check that the immediate updating strategy is rejected with workers."""
    with assert_exception(ValueError, snapshot):
        function()


def _build_without_database(settings_class, workers) -> None:
    """Build settings without database and with workers."""
    settings_class(use_database=False, workers=workers)


def _set_no_database_after_workers(settings_class, workers) -> None:
    """Disable the database of settings built with workers."""
    settings = settings_class(workers=workers)
    settings.use_database = False


@pytest.mark.parametrize("workers", [2, -1], ids=["two_workers", "all_the_cpus"])
@pytest.mark.parametrize(
    "function", [_build_without_database, _set_no_database_after_workers]
)
@pytest.mark.parametrize(
    "settings_class",
    [DIFFERENTIAL_EVOLUTION_Settings, SHGO_Settings],
    ids=["DIFFERENTIAL_EVOLUTION", "SHGO"],
)
def test_scipy_global_workers_without_database(
    settings_class, function, workers, snapshot
):
    """Check that the workers are rejected when the database is not used."""
    with assert_exception(ValueError, snapshot):
        function(settings_class, workers)


@pytest.mark.parametrize("workers", [0, -2], ids=["zero", "less_than_minus_one"])
@pytest.mark.parametrize(
    "settings_class",
    [DIFFERENTIAL_EVOLUTION_Settings, SHGO_Settings],
    ids=["DIFFERENTIAL_EVOLUTION", "SHGO"],
)
def test_scipy_global_rejects_invalid_workers(settings_class, workers, snapshot):
    """Check that the number of workers is positive or -1."""
    with assert_exception(ValidationError, snapshot):
        settings_class(workers=workers)


def test_shgo_accepts_all_the_cpus():
    """Check that SHGO accepts -1 workers, i.e. all the CPUs."""
    assert SHGO_Settings(workers=-1).workers == -1


_counting_settings = partial(
    DIFFERENTIAL_EVOLUTION_Settings,
    popsize=2,
    seed=1,
    tol=1e6,
    polish=False,
    max_iter=100,
)
"""Settings ending the run after the first generations.

The run does not end in the middle of a batch of evaluations:
the workers evaluate the points of a whole batch before stopping,
while the points read by SciPy are the only ones counted.
"""


def test_parallel_counts_like_serial(enable_function_statistics):
    """Check that the objective is called once per point, as in a serial run."""
    serial_problem = Rosenbrock()
    optimization_library_factory.execute(serial_problem, settings=_counting_settings())
    problem = Rosenbrock()
    optimization_library_factory.execute(
        problem, settings=_counting_settings(workers=2)
    )
    assert problem.objective.n_calls == serial_problem.objective.n_calls
    assert len(problem.database) == len(serial_problem.database)


def test_parallel_does_not_send_known_points():
    """Check that the workers get only the points missing from the database.

    A second run on the same problem finds every point in the database,
    so nothing is sent to the workers.
    """
    problem = Rosenbrock()
    optimization_library_factory.execute(
        problem, settings=_counting_settings(workers=2)
    )
    problem.database.clear_listeners()
    with mock.patch.object(
        ProcessPoolExecutor, "map", autospec=True, side_effect=ProcessPoolExecutor.map
    ) as map_:
        optimization_library_factory.execute(
            problem, settings=_counting_settings(workers=2)
        )

    assert map_.call_count == 0


def _create_problem_with_nan() -> OptimizationProblem:
    """Create a problem whose objective is NaN for a negative design variable.

    Returns:
        The problem.
    """
    design_space = DesignSpace()
    design_space.add_real_variable("x", lower_bound=-1.0, upper_bound=1.0, value=1.0)
    problem = OptimizationProblem(design_space)
    problem.objective = ArrayFunction(log, name="f")
    return problem


@pytest.mark.parametrize(
    ("workers", "in_calling_process"),
    [(1, False), (2, False), (2, True)],
    ids=["serial", "worker_processes", "calling_process"],
)
def test_parallel_stores_the_points_before_a_failure(
    request, workers, in_calling_process
):
    """Check that a failing point does not drop the points before it.

    A chunk of points sent to a worker used to be lost as a whole
    when one of its points raised,
    so the points before it in the chunk were not stored,
    unlike in a serial run.
    Here, the chunks have two points and the second one gives a NaN,
    which stops the run.
    """
    if in_calling_process:
        request.getfixturevalue("serial_executors")

    points = [array([0.5]), array([-0.5])] + [array([0.1 * i]) for i in range(1, 9)]

    def optimize(func, bounds, workers, **settings):
        if workers == 1:
            for point in points:
                func(point)
        else:
            workers(func, points)

    problem = _create_problem_with_nan()
    with mock.patch.object(
        ScipyGlobalOpt,
        "_ScipyGlobalOpt__names_to_functions",
        {"DIFFERENTIAL_EVOLUTION": optimize},
    ):
        optimization_library_factory.execute(
            problem,
            settings=DIFFERENTIAL_EVOLUTION_Settings(
                max_iter=100,
                workers=workers,
                updating="deferred",
                normalize_design_space=False,
            ),
        )

    assert_equal(
        problem.database.get_x_vect_history(),
        [array([1.0]), array([0.5]), array([-0.5])],
    )


class _SerialExecutor:
    """A pool of worker processes running its tasks in the calling process.

    The worker functions run on a copy of the library,
    as in a worker process,
    but where coverage measures them;
    the results are pickled and unpickled, as the ones of a real pool.
    """

    def __init__(self, max_workers, mp_context, initializer, initargs) -> None:
        self.max_workers = max_workers
        self.mapped_inputs = []
        initializer(*loads(dumps(initargs)))

    def map(self, function, inputs, chunksize):
        inputs = list(inputs)
        self.mapped_inputs.extend(task[0] for task in inputs)
        return (loads(dumps(function(loads(dumps(input_))))) for input_ in inputs)

    def shutdown(self, wait=True, cancel_futures=False) -> None:
        """Shut down the pool."""


@pytest.fixture
def serial_executors(monkeypatch) -> list[_SerialExecutor]:
    """The pools of worker processes created by `ScipyGlobalOpt`, run serially."""
    executors = []

    def create_executor(**kwargs):
        executor = _SerialExecutor(**kwargs)
        executors.append(executor)
        return executor

    monkeypatch.setattr(process_pool, "ProcessPoolExecutor", create_executor)
    monkeypatch.setattr(scipy_global, "_worker_library", None)
    # The calling process must not be tagged as a worker process.
    monkeypatch.setattr(scipy_global, "tag_process_worker", mock.Mock())
    return executors


def test_parallel_sends_a_repeated_point_once(serial_executors):
    """Check that a point repeated in a batch is sent once to the workers."""
    problem = Rosenbrock()
    point = array([0.5, 0.5])
    values = []

    def optimize(func, bounds, workers, **settings):
        values.extend(workers(func, [point, point]))
        return OptimizeResult(message="", success=True)

    with mock.patch.object(
        ScipyGlobalOpt,
        "_ScipyGlobalOpt__names_to_functions",
        {"DIFFERENTIAL_EVOLUTION": optimize},
    ):
        optimization_library_factory.execute(
            problem,
            settings=DIFFERENTIAL_EVOLUTION_Settings(
                max_iter=10, workers=2, normalize_design_space=False
            ),
        )

    assert_equal(serial_executors[0].mapped_inputs, [point])
    assert_equal(values, [6.5, 6.5])


def test_parallel_completes_a_known_point(serial_executors):
    """Check that a point of the database missing some outputs is completed.

    The known output is a sentinel that the objective cannot return:
    it must be read by the worker, not computed again.
    """
    problem = Power2()
    point = array([0.5, 0.5, 0.5])
    problem.database.store(point, {"pow2": array([123.0])})

    def optimize(func, bounds, workers, **settings):
        workers(func, [point])
        return OptimizeResult(message="", success=True)

    with mock.patch.object(
        ScipyGlobalOpt,
        "_ScipyGlobalOpt__names_to_functions",
        {"SHGO": optimize},
    ):
        optimization_library_factory.execute(
            problem,
            settings=SHGO_Settings(
                max_iter=10, workers=2, normalize_design_space=False
            ),
        )

    assert_equal(serial_executors[0].mapped_inputs, [point])
    outputs = problem.database.get(point)
    assert outputs.keys() == {"pow2", "ineq1", "ineq2", "eq"}
    assert_equal(outputs["pow2"], array([123.0]))


@pytest.mark.parametrize(
    ("platform_is_windows", "popsize", "n_processes"),
    [(False, 40, 64), (True, 40, 61), (False, 2, 4)],
    ids=["not_windows", "windows", "batch_size"],
)
def test_number_of_processes(
    serial_executors, monkeypatch, platform_is_windows, popsize, n_processes
):
    """Check that the number of worker processes is capped.

    `ProcessPoolExecutor` rejects more than 61 worker processes on Windows,
    and there is no need for more processes than points in a batch.
    """
    monkeypatch.setattr(scipy_global, "n_cpus", 64)
    monkeypatch.setattr(process_pool, "platform_is_windows", platform_is_windows)
    monkeypatch.setattr(
        start_method,
        "multi_processing_start_method",
        MultiProcessingStartMethod.SPAWN,
    )
    optimization_library_factory.execute(
        Rosenbrock(),
        settings=DIFFERENTIAL_EVOLUTION_Settings(
            max_iter=2, popsize=popsize, workers=-1
        ),
    )
    assert serial_executors[0].max_workers == n_processes


def test_differential_evolution_ignores_workers_with_constraints(
    serial_executors, caplog
):
    """Check that the workers of DIFFERENTIAL_EVOLUTION are not used with constraints.

    SciPy evaluates the constraints in the calling process,
    so the workers bring no speed-up.
    """
    caplog.set_level(logging.WARNING)
    optimization_library_factory.execute(
        Power2(),
        settings=DIFFERENTIAL_EVOLUTION_Settings(max_iter=3, popsize=2, workers=2),
    )
    assert "bring no speed-up with constraints" in caplog.text
    assert not serial_executors


def test_differential_evolution_default_updating_with_constraints():
    """Check that the default updating strategy follows the workers actually used.

    With constraints, the workers are dropped,
    so SciPy must receive the immediate updating strategy and 1 worker.
    """
    scipy_function = mock.Mock(wraps=differential_evolution)
    with mock.patch.object(
        ScipyGlobalOpt,
        "_ScipyGlobalOpt__names_to_functions",
        {"DIFFERENTIAL_EVOLUTION": scipy_function},
    ):
        optimization_library_factory.execute(
            Power2(),
            settings=DIFFERENTIAL_EVOLUTION_Settings(max_iter=3, popsize=2, workers=2),
        )

    kwargs = scipy_function.call_args.kwargs
    assert kwargs["updating"] == DifferentialEvolutionUpdating.IMMEDIATE
    assert kwargs["workers"] == 1


def test_parallel_counts_a_point_stored_with_no_output(serial_executors):
    """Check that a point stored with no output counts as a new point.

    An entry stored as an empty dictionary is a point to evaluate,
    as for the evaluation layer,
    so the iteration is counted.
    """
    point = array([0.5, 0.5])

    def optimize(func, bounds, workers, **settings):
        workers(func, [point])
        return OptimizeResult(message="", success=True)

    counters = []
    for stored_point in (None, point):
        problem = Rosenbrock()
        if stored_point is not None:
            problem.database.store(stored_point, {})

        with mock.patch.object(
            ScipyGlobalOpt,
            "_ScipyGlobalOpt__names_to_functions",
            {"DIFFERENTIAL_EVOLUTION": optimize},
        ):
            optimization_library_factory.execute(
                problem,
                settings=DIFFERENTIAL_EVOLUTION_Settings(
                    max_iter=10, workers=2, normalize_design_space=False
                ),
            )

        counters.append(problem.evaluation_counter.current)

    assert counters[0] == counters[1]


class _UnpicklableError(Exception):
    """An exception that cannot be pickled."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.lock = threading.Lock()


def _raise_if_negative(x):
    """Raise an unpicklable exception for a negative design variable."""
    if x[0] < 0:
        msg = "negative design variable"
        raise _UnpicklableError(msg)

    return x


def test_parallel_reports_an_unpicklable_exception(serial_executors):
    """Check that an unpicklable exception of a worker is not hidden.

    The pool would fail to send it back and raise another error instead,
    so the worker replaces it with a `RuntimeError`
    carrying its type name, its message and its traceback.
    The points before it are stored.
    """
    design_space = DesignSpace()
    design_space.add_real_variable("x", lower_bound=-1.0, upper_bound=1.0, value=1.0)
    problem = OptimizationProblem(design_space)
    problem.objective = ArrayFunction(_raise_if_negative, name="f")
    points = [array([0.5]), array([-0.5]), array([0.1])]

    def optimize(func, bounds, workers, **settings):
        workers(func, points)

    with (
        mock.patch.object(
            ScipyGlobalOpt,
            "_ScipyGlobalOpt__names_to_functions",
            {"DIFFERENTIAL_EVOLUTION": optimize},
        ),
        pytest.raises(
            RuntimeError,
            match=r"(?s)_UnpicklableError: negative design variable\n"
            r".*in _raise_if_negative",
        ),
    ):
        optimization_library_factory.execute(
            problem,
            settings=DIFFERENTIAL_EVOLUTION_Settings(
                max_iter=100,
                workers=2,
                updating="deferred",
                normalize_design_space=False,
            ),
        )

    assert_equal(problem.database.get_x_vect_history(), [array([1.0]), array([0.5])])


@pytest.fixture
def unconstrained_problem() -> OptimizationProblem:
    """An unconstrained optimization problem"""
    design_space = DesignSpace()
    design_space.add_real_variable("x", lower_bound=-1.0, upper_bound=1.0)
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
    design_space.add_integer_variable("x", lower_bound=-5, upper_bound=5, value=0)
    design_space.add_real_variable("y", lower_bound=-5.0, upper_bound=5.0, value=0.0)
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
