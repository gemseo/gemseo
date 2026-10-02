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
#                         documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Driver library tests."""

from __future__ import annotations

import gc
import logging
import weakref
from typing import TYPE_CHECKING
from typing import ClassVar
from unittest import mock

import pytest
from numpy import array
from numpy import full

from gemseo import configuration
from gemseo import execute_algo
from gemseo.core.algorithm import base_driver_library
from gemseo.core.algorithm._progress_bar.standard import ProgressBar
from gemseo.core.algorithm.base_driver_library import BaseDriverLibrary
from gemseo.core.function.array_function import ArrayFunction
from gemseo.core.function.collection.functions import Functions
from gemseo.core.function.linear_function import LinearFunction
from gemseo.doe.custom_doe.custom_doe import CustomDOE
from gemseo.doe.custom_doe.settings.custom_doe_settings import CustomDOE_Settings
from gemseo.doe.scipy.scipy_doe import SciPyDOE
from gemseo.doe.scipy.settings.mc import MC_Settings
from gemseo.optimization.factory import optimization_library_factory
from gemseo.optimization.problem import OptimizationProblem
from gemseo.optimization.result import OptimizationResult
from gemseo.optimization.scipy_local.scipy_local import ScipyOpt
from gemseo.optimization.scipy_local.settings.lbfgsb import L_BFGS_B_Settings
from gemseo.optimization.scipy_local.settings.slsqp import SLSQP_Settings
from gemseo.problem.optimization.power_2 import Power2
from gemseo.problem.optimization.rosenbrock import Rosenbrock
from gemseo.space._core.rendering import render_string
from gemseo.space.design import DesignSpace
from gemseo.space.util import get_value_and_bounds
from gemseo.util.pydantic import create_model
from gemseo.util.testing.helper import assert_exception
from gemseo.util.testing.helper import concretize_classes

if TYPE_CHECKING:
    from gemseo.core.algorithm.base_driver_library import DriverDescription


@pytest.fixture(scope="module")
def power_2() -> Power2:
    """The power-2 problem."""
    problem = Power2()
    problem.database.store(
        array([0.79499653, 0.20792012, 0.96630481]),
        {"pow2": 1.61, "ineq1": -0.0024533, "ineq2": -0.0024533, "eq": -0.00228228},
    )
    return problem


class MyDriver(BaseDriverLibrary):
    ALGORITHM_INFOS: ClassVar[dict[str, DriverDescription]] = {"algo_name": None}

    def __init__(self, algo_name: str = "algo_name") -> None:
        super().__init__(algo_name)


@pytest.fixture(scope="module")
def optimization_problem():
    """A mock optimization problem."""
    design_space = mock.Mock()
    design_space.dimension = 2
    problem = mock.Mock()
    problem.dimension = 2
    problem.input_space = design_space
    problem.functions = Functions()
    return problem


def test_empty_design_space(snapshot) -> None:
    """Check that a driver cannot be executed with an empty design space."""
    with concretize_classes(MyDriver):
        driver = MyDriver()
    driver._algo_name = "algo_name"
    with assert_exception(ValueError, snapshot):
        driver._check_algorithm(OptimizationProblem(DesignSpace()))


def test_no_functions(optimization_problem, snapshot):
    """Check that an error is raised when the problem has no function."""
    lib = ScipyOpt("SLSQP")
    with assert_exception(ValueError, snapshot):
        lib.execute(optimization_problem)


@pytest.mark.parametrize("enable_progress_bar", [False, True])
@pytest.mark.parametrize("enable_logging", [False, True])
def test_progress_bar(enable_progress_bar, enable_logging, caplog) -> None:
    """Check the activation of the progress bar from the options of a
    BaseDriverLibrary."""
    enable = configuration.logging.enable
    configuration.logging.enable = enable_logging
    driver = optimization_library_factory.create("SLSQP")
    driver.execute(
        Power2(), settings=SLSQP_Settings(enable_progress_bar=enable_progress_bar)
    )
    use_progress_bar = enable_logging and enable_progress_bar
    assert isinstance(driver._progress_bar, ProgressBar) is use_progress_bar
    assert (
        "gemseo.core.algorithm._progress_bar.custom:custom.py:" in caplog.text
    ) is use_progress_bar
    configuration.logging.enable = enable


@pytest.mark.parametrize(
    ("kwargs", "expected"), [({}, "    50%|"), ({"message": "foo"}, "foo  50%|")]
)
def test_progress_bar_update(caplog, kwargs, expected) -> None:
    """Check the update of the progress bar when finalizing an iteration."""
    power_2 = Power2()
    test_driver = ScipyOpt("SLSQP")
    test_driver._problem = power_2
    test_driver._settings = create_model(
        test_driver.ALGORITHM_INFOS[test_driver.algo_name].settings_class
    )
    test_driver._settings.max_time = 0
    test_driver._init_iter_observer(power_2, max_iter=2, **kwargs)
    test_driver._problem.bind_functions()
    for function in test_driver._problem.functions:
        function.pre_compute_at_new_point = (
            test_driver._finalize_previous_iteration_using_database
        )
    test_driver._problem.evaluate_functions(
        array([0.0, 0.0, 0.0]),
        input_value_is_normalized=False,
        preprocess_input_value=False,
    )
    test_driver._problem.evaluate_functions(
        array([1.0, 0.0, 0.0]),
        input_value_is_normalized=False,
        preprocess_input_value=False,
    )
    assert expected in caplog.text

    # The driver library is not used through its execute method,
    # which is in charge of closing the progress bar
    # and of breaking the reference cycle
    # test_driver -> power_2 -> pre_compute_at_new_point -> test_driver.
    for function in test_driver._problem.functions:
        function.pre_compute_at_new_point = None

    test_driver._progress_bar.close()


@pytest.fixture
def driver_library() -> BaseDriverLibrary:
    """A driver library."""
    driver_library = ScipyOpt("SLSQP")
    design_space = DesignSpace()
    design_space.add_real_variable("x", lower_bound=-2.0, upper_bound=3.0, value=1.0)
    driver_library._problem = OptimizationProblem(design_space)
    return driver_library


@pytest.mark.parametrize(
    ("as_dict", "x0", "lower_bounds", "upper_bounds"),
    [(False, 1, -2, 3), (True, {"x": 1}, {"x": -2}, {"x": 3})],
)
def test_get_value_and_bounds_vects(
    driver_library, as_dict, x0, lower_bounds, upper_bounds
) -> None:
    """Check the getting of the initial values and bounds."""
    assert get_value_and_bounds(
        driver_library._problem.input_space, as_dict=as_dict
    ) == (
        x0,
        lower_bounds,
        upper_bounds,
    )


@pytest.mark.parametrize("name", ["new_iter_listener", "store_listener"])
def test_clear_listeners(name):
    """Check clear_listeners."""
    problem = Power2()
    getattr(problem.database, f"add_{name}")(sum)
    driver = CustomDOE()
    driver.execute(
        problem, settings=CustomDOE_Settings(samples=array([[-0.5, 0.0, 0.5]]))
    )
    assert getattr(problem.database, f"_Database__{name}s") == [sum]


@pytest.mark.parametrize("max_dimension", [1, 3])
def test_max_input_space_dimension_to_log(max_dimension, caplog):
    """Check the cap on the dimension of a design space to log."""
    problem = Power2()
    table = render_string(problem.input_space, use_html=False).split("\n", 1)[1]
    initial_space_string = "   over the design space:\n      " + table.replace(
        "\n", "\n      "
    )
    CustomDOE().execute(
        problem,
        settings=CustomDOE_Settings(
            samples=full((1, 3), pow(0.9, 1.0 / 3.0)),
            max_input_space_dimension_to_log=max_dimension,
        ),
    )

    # Check the logging of the initial design space
    assert (max_dimension >= 3) == (
        (
            "gemseo.core.algorithm.base_driver_library",
            logging.INFO,
            initial_space_string,
        )
        in caplog.record_tuples
    )

    # Check the logging of the final design space
    assert (max_dimension >= 3) == (
        (
            "gemseo.core.algorithm.base_driver_library",
            logging.INFO,
            render_string(problem.input_space, use_html=False)
            .replace("Design space", "      Design space")
            .replace("\n", "\n         "),
        )
        in caplog.record_tuples
    )


class MockedTime:
    """Mock time, returning 0 at first call, 10 at second call and 10 at other calls."""

    def __init__(self):
        self.n_calls = 0

    def __call__(self, *args, **kwargs):
        self.n_calls += 1
        if self.n_calls in [1, 2]:
            return 0.0

        return 10


def test_reaching_max_time_does_not_stop_storing():
    """Check that reaching maximum time does not stop storing in the database."""
    problem = Rosenbrock()
    problem.add_constraint(
        ArrayFunction(sum, name="sum"), constraint_type=ArrayFunction.ConstraintType.EQ
    )
    n_samples = 100
    with mock.patch.object(base_driver_library, "time", MockedTime()):
        SciPyDOE("MC").execute(
            problem, settings=MC_Settings(n_samples=n_samples, max_time=1)
        )

    # Reaching maximum time stops iterating.
    assert len(problem.database) < n_samples
    # Reaching maximum time does not stop storing in the database.
    assert len(problem.database.last_item) == 2


class G:
    def __init__(self):
        self.value = 1.0

    def __call__(self, x):
        self.value *= -1
        return self.value


@pytest.mark.parametrize("use_database", [True, False])
@pytest.mark.parametrize("n_processes", [1, 2])
def test_progress_bar_database_n_processes(caplog, use_database, n_processes):
    """Check that the progress bar is logged w/wo parallelization and w/wo database."""
    problem = Rosenbrock()
    problem.add_constraint(
        ArrayFunction(G(), name="g"),
        value=1.0,
        constraint_type=problem.ConstraintType.EQ,
    )
    execute_algo(
        problem,
        "doe",
        settings_model=MC_Settings(
            n_samples=3, n_processes=n_processes, use_database=use_database
        ),
    )
    assert "33%" in caplog.text
    assert ("obj=325" in caplog.text) is use_database
    assert "67%" in caplog.text
    assert ("obj=11.2" in caplog.text) is use_database
    assert "100%" in caplog.text
    assert ("obj=79.3" in caplog.text) is use_database


def test_get_result_without_result_class(snapshot) -> None:
    """Check that _get_result raises when _result_class is not set."""
    with concretize_classes(MyDriver):
        driver = MyDriver()
    with assert_exception(NotImplementedError, snapshot):
        driver._get_result(OptimizationProblem(DesignSpace()), "message", None)


def test_reset_releases_the_problem_of_the_run() -> None:
    """Check that a reset drops the problem the user built and the map to it.

    These two are the state of a run that this reset owns,
    on top of the problem and the settings the base class clears.
    A driver library is serializable,
    so anything left behind is dragged into a pickle taken after the run.

    Per-run state that this reset does not own is out of the scope of this test:
    a driver may still hold the problem,
    or parts of it,
    through its progress bar
    or through the functions a DOE library keeps,
    which is why the progress bar is disabled here,
    together with the logging of the problem,
    since a captured log record holds the object it formats.
    """
    driver = ScipyOpt("SLSQP")
    problem = Power2()
    driver.execute(
        problem,
        settings=SLSQP_Settings(
            max_iter=2, enable_progress_bar=False, log_problem=False
        ),
    )

    assert driver._original_problem is None
    assert driver._transformation is None

    reference = weakref.ref(problem)
    del problem
    gc.collect()
    assert reference() is None


def test_reset_releases_the_state_of_a_run_that_raises(snapshot) -> None:
    """Check that a reset drops the state of a run the algorithm did not finish.

    An algorithm raising leaves as much behind as an algorithm returning,
    so the state of a run is cleared
    whatever the outcome of that run.
    """

    def raise_an_error(input_value):
        msg = "The objective cannot be evaluated."
        raise RuntimeError(msg)

    design_space = DesignSpace()
    design_space.add_real_variable("x", lower_bound=0.0, upper_bound=1.0, value=0.5)
    problem = OptimizationProblem(design_space)
    problem.objective = ArrayFunction(raise_an_error, name="f")
    driver = ScipyOpt("SLSQP")
    with assert_exception(RuntimeError, snapshot):
        driver.execute(problem, settings=SLSQP_Settings(enable_progress_bar=False))

    assert driver._problem is None
    assert driver._original_problem is None
    assert driver._settings is None
    assert driver._transformation is None


def test_the_hooks_of_a_run_that_raises_are_released(snapshot) -> None:
    """Check the evaluation layer of a problem an algorithm raised on.

    The hooks a driver sets on that layer are its own,
    so it releases them
    whatever the outcome of the run.
    Left in place,
    the one finalizing an iteration would keep the driver alive
    through the functions of the user
    and fire on the next run.
    """

    def raise_an_error(input_value):
        msg = "The objective cannot be evaluated."
        raise RuntimeError(msg)

    design_space = DesignSpace()
    design_space.add_real_variable("x", lower_bound=0.0, upper_bound=1.0, value=0.5)
    problem = OptimizationProblem(design_space)
    problem.objective = ArrayFunction(raise_an_error, name="f")
    problem.add_observable(ArrayFunction(sum, name="o"), new_iter=True)
    driver = ScipyOpt("SLSQP")
    with assert_exception(RuntimeError, snapshot):
        driver.execute(problem, settings=SLSQP_Settings(enable_progress_bar=False))

    assert problem.database._Database__new_iter_listeners == []
    for function in problem.functions:
        assert function.pre_compute_at_new_point is None


_normalize_design_space_ignored_message = (
    "The setting normalize_design_space is ignored"
)


def test_normalize_design_space_explicitly_set_warns_on_augmented_lagrangian(
    caplog,
) -> None:
    """Check the warning on a library not iterating on a working problem.

    The augmented Lagrangian delegates to a sub-algorithm,
    which normalizes according to its own settings,
    so `normalize_design_space` has no effect at the top level;
    an explicit request for it is worth a warning.
    """
    execute_algo(
        Power2(),
        algo_name="Augmented_Lagrangian_Order_1",
        max_iter=10,
        normalize_design_space=True,
        sub_algorithm_settings=L_BFGS_B_Settings(),
    )

    assert _normalize_design_space_ignored_message in caplog.text


def test_normalize_design_space_default_is_quiet_on_augmented_lagrangian(
    caplog,
) -> None:
    """Check that a run passing nothing explicitly does not warn.

    The augmented Lagrangian settings default `normalize_design_space` to
    `True`, the same value a run leaves untouched,
    so a default run stays quiet:
    only an explicit request warns.
    """
    execute_algo(
        Power2(),
        algo_name="Augmented_Lagrangian_Order_1",
        max_iter=10,
        sub_algorithm_settings=L_BFGS_B_Settings(),
    )

    assert _normalize_design_space_ignored_message not in caplog.text


def test_normalize_design_space_explicit_false_is_quiet_on_augmented_lagrangian(
    caplog,
) -> None:
    """Check that an explicit `False` does not warn either.

    Only an explicit `True` has no effect worth a warning about;
    `False` asks for what the library already does.
    """
    execute_algo(
        Power2(),
        algo_name="Augmented_Lagrangian_Order_1",
        max_iter=10,
        normalize_design_space=False,
        sub_algorithm_settings=L_BFGS_B_Settings(),
    )

    assert _normalize_design_space_ignored_message not in caplog.text


def _milp_problem() -> OptimizationProblem:
    """A minimal linear problem, small enough for a MILP solver to run fast.

    Returns:
        The problem.
    """
    design_space = DesignSpace()
    design_space.add_real_variable("x", lower_bound=0.0, upper_bound=1.0, value=1.0)
    problem = OptimizationProblem(design_space)
    problem.objective = LinearFunction(
        array([1.0]), "f", ArrayFunction.FunctionType.OBJ, ["x"]
    )
    return problem


def test_projected_optimum_evaluation_is_not_recorded() -> None:
    """Check that evaluating the projected optimum does not record it.

    `__set_projected_optimum` runs after the result is built, so recording
    that evaluation in the problem's database would misalign it with
    `result.x_opt`: `problem.database` would then hold a point the algorithm
    never actually visited, e.g. showing up as an extra iteration in a
    history plot or after a round trip through HDF.
    """
    design_space = DesignSpace()
    design_space.add_real_variable("x", lower_bound=0.0, upper_bound=10.0, value=1.0)
    problem = OptimizationProblem(design_space)
    problem.objective = ArrayFunction(lambda x: array([(x[0] - 3.0) ** 2]), name="f")
    problem.add_constraint(
        ArrayFunction(lambda x: array([x[0] - 8.0]), name="g"),
        constraint_type=ArrayFunction.ConstraintType.INEQ,
    )
    problem.bind_functions()
    problem.evaluate_functions(
        input_value=array([3.0]), input_value_is_normalized=False
    )
    n_entries_before_projection = len(problem.database)

    driver = ScipyOpt("SLSQP")
    result = OptimizationResult(x_opt=array([3.0]), f_opt=array([0.0]))
    driver._BaseDriverLibrary__set_projected_optimum(problem, result, array([4.0]))

    assert len(problem.database) == n_entries_before_projection
    assert result.f_opt_projected == array([1.0])
    assert result.is_feasible_projected


def test_projected_optimum_of_a_maximization_problem() -> None:
    """Check the projected optimum of a maximization problem with a relaxed integer.

    `f_opt_projected` is negated back the same way as `f_opt` is in
    `OptimizationResult.from_optimization_problem`
    when the problem both maximizes the objective
    and does not use the standardized one,
    the one branch of `__set_projected_optimum` no other test covers.
    """
    design_space = DesignSpace()
    design_space.add_integer_variable("x", lower_bound=0, upper_bound=10, value=5)
    problem = OptimizationProblem(design_space)
    problem.objective = ArrayFunction(
        lambda x: array([-((x[0] - 3.4) ** 2)]),
        name="f",
        jac=lambda x: array([[-2.0 * (x[0] - 3.4)]]),
    )
    problem.minimize_objective = False
    problem.use_standardized_objective = False

    result = ScipyOpt("SLSQP").execute(
        problem, settings=SLSQP_Settings(relax_integer_variables=True, max_iter=20)
    )

    # SLSQP explores "x" continuously and settles close to 3.4,
    # which the projection onto the declared, integer-only domain rounds to 3,
    # moving the optimum.
    assert result.x_opt == pytest.approx(array([3.4]), abs=0.1)
    assert result.x_opt_projected == array([3.0])
    assert result.f_opt_projected == pytest.approx(array([-0.16]))


def test_projected_optimum_with_a_failing_observable() -> None:
    """Check the projected optimum when an observable fails at that point.

    `__set_projected_optimum` evaluates the objective and the constraints of
    the problem, but not its observables, at the point projected onto the
    declared domain: an observable failing there, e.g. one undefined off the
    domain the algorithm explored, must not prevent `f_opt_projected` and
    `is_feasible_projected` from being set.
    """
    design_space = DesignSpace()
    design_space.add_integer_variable("x", lower_bound=0, upper_bound=3, value=3)
    problem = OptimizationProblem(design_space)
    problem.objective = ArrayFunction(
        lambda x: array([(x[0] - 1.4) ** 2]),
        name="f",
        jac=lambda x: array([[2.0 * (x[0] - 1.4)]]),
    )

    def _observable(x):
        if float(x[0]).is_integer() and x[0] < 2.5:
            msg = "The observable fails at integer points close to the optimum."
            raise RuntimeError(msg)
        return x

    problem.add_observable(ArrayFunction(_observable, name="o"))

    result = ScipyOpt("SLSQP").execute(
        problem, settings=SLSQP_Settings(relax_integer_variables=True, max_iter=20)
    )

    assert result.x_opt_projected == array([1.0])
    assert result.f_opt_projected == pytest.approx(array([0.16]))
    assert result.is_feasible_projected


def test_normalize_design_space_explicitly_set_warns_on_scipy_milp(caplog) -> None:
    """Check the warning on `ScipyMILP`, which does not iterate on a working problem."""
    execute_algo(_milp_problem(), algo_name="MILP", normalize_design_space=True)

    assert _normalize_design_space_ignored_message in caplog.text


def test_normalize_design_space_default_is_quiet_on_scipy_milp(caplog) -> None:
    """Check that a `ScipyMILP` run passing nothing explicitly does not warn."""
    execute_algo(_milp_problem(), algo_name="MILP")

    assert _normalize_design_space_ignored_message not in caplog.text


def test_identity_transformation_still_builds_the_working_problem() -> None:
    """Check that an identity transformation still builds a working problem.

    Skipping it, as used to be done, would leave the algorithm iterating on
    the problem the user built itself, e.g. for a default DOE or an
    optimizer with `normalize_design_space` set to `False` and nothing to
    relax; an algorithm mutating the problem it iterates on, such as
    `BaseOptimizationLibrary._pre_run` scaling the objective and the
    constraints, would then mutate the user's.
    """
    problem = Power2()
    with mock.patch.object(
        problem, "create_working_problem", wraps=problem.create_working_problem
    ) as mock_create_working_problem:
        execute_algo(
            problem,
            algo_type="doe",
            settings_model=CustomDOE_Settings(samples=array([[0.5, 0.5, 0.5]])),
        )

    mock_create_working_problem.assert_called_once()


def test_non_identity_transformation_still_builds_the_working_problem() -> None:
    """Check that a non-identity transformation still builds a working problem."""
    problem = Power2()
    with mock.patch.object(
        problem, "create_working_problem", wraps=problem.create_working_problem
    ) as mock_create_working_problem:
        execute_algo(problem, algo_name="SLSQP", max_iter=1)

    mock_create_working_problem.assert_called_once()


def test_empty_transformation_does_not_mutate_the_users_problem() -> None:
    """Check that scaling under an empty transformation leaves the user's problem be.

    `normalize_design_space` set to `False` and nothing to relax builds an
    empty transformation. `BaseOptimizationLibrary._pre_run` still replaces
    the objective and the constraints with scaled versions when
    `scaling_threshold` is set, and, the working problem being built all the
    same, this lands on it, not on the problem the user built: its objective
    is left as it was after the run, and a second run does not keep
    recording into its database through the first run's scaled functions.
    """
    problem = Rosenbrock(initial_guess=array([-1.5, 1.5]))
    x_0 = problem.design_space.get_current_value().copy()

    execute_algo(
        problem,
        algo_name="L_BFGS_B",
        max_iter=5,
        scaling_threshold=1.0,
        normalize_design_space=False,
    )

    assert problem.objective.evaluate(x_0) == pytest.approx(62.5)

    problem.database.clear()
    problem.design_space.set_current_value(x_0)
    execute_algo(
        problem,
        algo_name="L_BFGS_B",
        max_iter=5,
        scaling_threshold=1.0,
        normalize_design_space=False,
        use_database=False,
    )

    assert len(problem.database) == 0
