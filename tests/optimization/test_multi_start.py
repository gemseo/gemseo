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
from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import pytest
from numpy import allclose
from numpy import array
from numpy import nan
from numpy import vstack
from numpy.testing import assert_almost_equal

from gemseo.core.function.array_function import ArrayFunction
from gemseo.core.problem.database import Database
from gemseo.doe.custom_doe.settings.custom_doe_settings import CustomDOE_Settings
from gemseo.doe.scipy.settings.lhs import LHS_Settings
from gemseo.doe.scipy.settings.mc import MC_Settings
from gemseo.optimization.factory import optimization_library_factory
from gemseo.optimization.multi_start.multi_start import MultiStart
from gemseo.optimization.multi_start.settings.multi_start_settings import (
    MultiStart_Settings,
)
from gemseo.optimization.nlopt.settings.nlopt_cobyla_settings import (
    NLOPT_COBYLA_Settings,
)
from gemseo.optimization.problem import OptimizationProblem
from gemseo.optimization.scipy_local.settings.slsqp import SLSQP_Settings
from gemseo.problem.optimization.power_2 import Power2
from gemseo.space.design import DesignSpace
from gemseo.util.testing.helper import assert_exception

if TYPE_CHECKING:
    from gemseo.util.typing import RealArray


class _FailingPower2(Power2):
    """A `Power2` problem whose objective can be made to fail on demand.

    The objective fails either when `x[0]` is below a threshold
    (deterministic and x-based, unlike `Power2.exception_error`),
    or on a given objective call index, counting from 1
    (deterministic only when `n_processes` is 1,
    since each process then has its own call counter;
    this counter is shared by all the sub-optimizations of a run,
    as the objective belongs to the parent problem).
    It is defined at module level so that it can be pickled by qualified name,
    which is required for the multiprocessing tests.
    """

    evaluated_x: list[RealArray]
    """The design values passed to the objective on every successful call."""

    failing_x: RealArray | None
    """The design value passed to the objective on the failing call, if any."""

    def __init__(
        self,
        mode: str = "exception",
        threshold: float = 0.25,
        failing_call_index: int = 0,
    ) -> None:
        """
        Args:
            mode: The way the objective fails,
                either by raising a `ValueError` (`"exception"`)
                or by returning a NaN value (`"nan"`).
            threshold: The value of `x[0]` below which the objective fails,
                used only when `failing_call_index` is `0`.
            failing_call_index: The 1-based index of the objective call that
                fails. When `0`, the objective fails instead when `x[0]` is
                below `threshold`.
        """  # noqa: D205 D212
        self._mode = mode
        self._threshold = threshold
        self._failing_call_index = failing_call_index
        self._n_calls = 0
        self.evaluated_x = []
        self.failing_x = None
        super().__init__()

    def pow2(self, x_dv: RealArray) -> RealArray:
        """Compute the objective, failing according to the configured rule.

        Args:
            x_dv: The design variable vector.

        Returns:
            The objective value, or a NaN value in the `"nan"` failure mode.

        Raises:
            ValueError: In the `"exception"` failure mode,
                when the failure condition is met.
        """
        self._n_calls += 1
        if self._failing_call_index:
            fails = self._n_calls == self._failing_call_index
        else:
            fails = x_dv[0] < self._threshold

        if fails:
            self.failing_x = x_dv.copy()
            if self._mode == "exception":
                msg = "The discipline diverged."
                raise ValueError(msg)
            return array([nan])

        self.evaluated_x.append(x_dv.copy())
        return super().pow2(x_dv)


@pytest.mark.parametrize(
    ("max_iter", "settings", "expected_length"),
    [
        (10, {"doe_algo_settings": LHS_Settings(n_samples=5)}, 10),
        (5, {"doe_algo_settings": LHS_Settings(n_samples=4)}, 5),
        (
            15,
            {
                "opt_algo_settings": SLSQP_Settings(max_iter=2),
                "doe_algo_settings": LHS_Settings(n_samples=5),
            },
            11,
        ),
    ],
)
def test_database_length(
    max_iter, settings, expected_length, enable_function_statistics
):
    """Check the database length and the number of calls to the objective.

    Flaky test, convergence is dependent on the SLSQP build.
    """
    problem = Power2()
    algo = MultiStart()
    algo.execute(problem, settings=MultiStart_Settings(max_iter=max_iter, **settings))
    assert len(problem.database) == expected_length
    assert problem.objective.n_calls == 1


@pytest.mark.parametrize("max_iter", [4, 5])
def test_max_iter_error_1(max_iter, snapshot):
    """Check that max_iter <= n_start raises an error."""
    problem = Power2()
    algo = MultiStart()
    with assert_exception(ValueError, snapshot):
        algo.execute(
            problem,
            settings=MultiStart_Settings(
                max_iter=max_iter, doe_algo_settings=LHS_Settings(n_samples=5)
            ),
        )


def test_max_iter_error_2(snapshot):
    """Check that opt_algo_max_iter * n_start > max_iter raises an error."""
    problem = Power2()
    algo = MultiStart()
    with assert_exception(ValueError, snapshot):
        algo.execute(
            problem,
            settings=MultiStart_Settings(
                max_iter=10,
                opt_algo_settings=SLSQP_Settings(max_iter=10),
                doe_algo_settings=LHS_Settings(n_samples=5),
            ),
        )


@pytest.mark.parametrize("n_processes", [1, 2])
def test_database(n_processes):
    """Check that the initial points are in the database."""
    problem = Power2()
    algo = MultiStart()
    algo.execute(
        problem,
        settings=MultiStart_Settings(
            max_iter=10,
            doe_algo_settings=CustomDOE_Settings(
                samples=array([[0.2, 0.5, 0.4], [0.3, 0.8, 0.2]])
            ),
            n_processes=n_processes,
        ),
    )
    x_history = vstack(problem.database.get_x_vect_history())
    # The first iteration is the evaluation of the functions
    # at the initial design value.
    assert_almost_equal(
        x_history[[0, 1, 6]], array([[1.0, 1.0, 1.0], [0.2, 0.5, 0.4], [0.3, 0.8, 0.2]])
    )


def test_normalize_design_space():
    """Check that normalizing the design space changes nothing but the coordinates.

    This algorithm builds sub-problems pairing a copy of the input space
    with the original functions,
    so it must be handed the problem the user declared;
    iterating on a working one paired the normalized space
    with the functions of the original one
    and returned a wrong optimum.
    """
    x_opts = []
    for normalize_design_space in (False, True):
        design_space = DesignSpace()
        # A range straddling neither 0 nor 1,
        # so that a point of the normalized space is outside the declared bounds.
        design_space.add_real_variable(
            "x", lower_bound=10.0, upper_bound=20.0, value=12.0
        )
        problem = OptimizationProblem(design_space)
        problem.objective = ArrayFunction(
            lambda x: array([(x[0] - 16.0) ** 2]),
            name="f",
            jac=lambda x: array([[2.0 * (x[0] - 16.0)]]),
        )
        result = MultiStart().execute(
            problem,
            settings=MultiStart_Settings(
                max_iter=20,
                normalize_design_space=normalize_design_space,
                opt_algo_settings=SLSQP_Settings(max_iter=5),
                doe_algo_settings=CustomDOE_Settings(samples=array([[11.0], [19.0]])),
            ),
        )
        x_opts.append(result.x_opt)
        for x_vect in problem.database.get_x_vect_history():
            assert 10.0 <= x_vect[0] <= 20.0

    assert_almost_equal(x_opts[1], x_opts[0])


def test_factory():
    """Check that the factory of optimization algorithms knows this algorithm."""
    assert optimization_library_factory.is_available("MultiStart")


def test_relaxed_integer_variable():
    """Check that a relaxed integer variable is projected back and flagged.

    `MultiStart` iterates on no working problem of its own,
    so its own `_transformation` is an empty composition;
    the point it writes back into the design space, and `to_dataset()`,
    must nonetheless account for the relaxation the sub-optimizations did.
    """
    design_space = DesignSpace()
    design_space.add_integer_variable("x", lower_bound=0, upper_bound=10)
    problem = OptimizationProblem(design_space)
    problem.objective = ArrayFunction(
        lambda x: array([(x[0] - 3.4) ** 2]),
        name="f",
        jac=lambda x: array([[2.0 * (x[0] - 3.4)]]),
    )
    algo = MultiStart()
    algo.execute(
        problem,
        settings=MultiStart_Settings(
            max_iter=20,
            opt_algo_settings=SLSQP_Settings(relax_integer_variables=True, max_iter=5),
            doe_algo_settings=CustomDOE_Settings(samples=array([[1.0], [5.0]])),
        ),
    )
    assert design_space.get_current_value() == array([3])
    assert "x" in problem.database.relaxed_variable_names
    # `to_dataset` casts an integer input column to a `pandas.Int64Dtype`,
    # which fails on a relaxed, non-integral, value unless the column is
    # exported as a float one, decided from `relaxed_variable_names`.
    dataset = problem.to_dataset()
    assert dataset.get_view(variable_names="x").to_numpy().dtype == float


@pytest.fixture(scope="module")
def x_history() -> RealArray:
    """The reference samples."""
    problem = Power2()
    algo = MultiStart()
    algo.execute(
        problem,
        settings=MultiStart_Settings(
            max_iter=10, doe_algo_settings=LHS_Settings(n_samples=5)
        ),
    )
    return vstack(problem.database.get_x_vect_history())


@pytest.mark.parametrize(
    "settings",
    [
        {"opt_algo_settings": NLOPT_COBYLA_Settings()},
        {"doe_algo_settings": MC_Settings(n_samples=5)},
    ],
)
def test_algo_settings(x_history, settings):
    """Check that the algorithm settings can be changed."""
    problem = Power2()
    algo = MultiStart()
    algo.execute(problem, settings=MultiStart_Settings(max_iter=10, **settings))
    assert not allclose(vstack(problem.database.get_x_vect_history()), x_history)


@pytest.fixture(scope="module")
def x_history_cobyla() -> RealArray:
    """The reference samples with COBYLA algorithm."""
    problem = Power2()
    algo = MultiStart()
    algo.execute(
        problem,
        settings=MultiStart_Settings(
            max_iter=10,
            opt_algo_settings=NLOPT_COBYLA_Settings(),
            doe_algo_settings=LHS_Settings(n_samples=5),
        ),
    )
    return vstack(problem.database.get_x_vect_history())


@pytest.mark.parametrize(
    "settings",
    [
        {
            "opt_algo_settings": NLOPT_COBYLA_Settings(init_step=0.5),
            "doe_algo_settings": LHS_Settings(n_samples=5),
        },
        {"doe_algo_settings": LHS_Settings(n_samples=5, scramble=False)},
    ],
)
def test_algo_settings_(x_history_cobyla, settings):
    """Check that the algorithm settings can be changed."""
    problem = Power2()
    algo = MultiStart()
    kwargs = {"opt_algo_settings": NLOPT_COBYLA_Settings()}
    kwargs.update(settings)
    algo.execute(problem, settings=MultiStart_Settings(max_iter=10, **kwargs))
    assert not allclose(vstack(problem.database.get_x_vect_history()), x_history_cobyla)


def test_multistart_file_path(tmp_wd):
    """Check the multistart_file_path option."""
    problem = Power2()
    algo = MultiStart()
    algo.execute(
        problem,
        settings=MultiStart_Settings(
            max_iter=10,
            doe_algo_settings=LHS_Settings(n_samples=5),
            multistart_file_path="local_optima.hdf5",
        ),
    )
    problem = problem.__class__.from_hdf("local_optima.hdf5")
    assert len(problem.database) == 5


@pytest.mark.parametrize("n_processes", [1, 2])
def test_skip_failed_starting_points(n_processes, caplog):
    """Check that a failed starting point is skipped while others still run.

    The logged failure reason is also checked.
    """
    problem = _FailingPower2()
    algo = MultiStart()
    algo.execute(
        problem,
        settings=MultiStart_Settings(
            max_iter=10,
            doe_algo_settings=CustomDOE_Settings(
                samples=array([[0.1, 0.5, 0.4], [0.5, 0.5, 0.9]])
            ),
            skip_failed_starting_points=True,
            n_processes=n_processes,
        ),
    )
    assert "skipping the starting point 1" in caplog.text
    assert "WARNING" in caplog.text
    assert "ValueError: The discipline diverged." in caplog.text
    x_history = problem.database.get_x_vect_history()
    assert any(allclose(x, [0.5, 0.5, 0.9]) for x in x_history)
    assert problem.solution.x_opt is not None


@pytest.mark.parametrize("n_processes", [1, 2])
def test_skip_failed_starting_points_disabled(snapshot, n_processes):
    """Check that disabling the skipping stops on the first failure."""
    problem = _FailingPower2()
    algo = MultiStart()
    with assert_exception(ValueError, snapshot):
        algo.execute(
            problem,
            settings=MultiStart_Settings(
                max_iter=10,
                doe_algo_settings=CustomDOE_Settings(
                    samples=array([[0.1, 0.5, 0.4], [0.5, 0.5, 0.9]])
                ),
                skip_failed_starting_points=False,
                n_processes=n_processes,
            ),
        )


def test_skip_failed_starting_point_worker_returns_none(monkeypatch, caplog):
    """Check that a worker returning `None` is treated as a failed starting point.

    In parallel mode, `CallableParallelExecution.execute` is called without
    `exceptions_to_re_raise`, so a worker failure occurring outside
    `_optimize`'s own `try`/`except` (e.g. a returned `OptimizationProblem`
    that cannot be pickled back to the parent process, or a worker process
    that dies abruptly, making the pool raise a `BrokenProcessPool`) is
    swallowed and reported as a `None` entry in the list of results, rather
    than as a `(sub_problem, error)` pair. `execute` is monkeypatched to
    emulate this, since triggering such a low-level worker failure directly
    is impractical.
    """
    problem = Power2()
    algo = MultiStart()

    def fake_execute(worker, callbacks, n_processes, inputs, exceptions_to_re_raise=()):
        """Emulate a worker crash by returning `None` for the first input.

        Args:
            worker: The callable performing the sub-optimizations.
            callbacks: The callback functions (unused, kept for signature parity).
            n_processes: The number of processes (unused, kept for signature parity).
            inputs: The inputs to the worker.
            exceptions_to_re_raise: The exception types to re-raise (unused,
                kept for signature parity).

        Returns:
            The outputs of the sub-optimizations, `None` for the first one.
        """
        del callbacks, n_processes, exceptions_to_re_raise
        return [None, *(worker(input_) for input_ in inputs[1:])]

    monkeypatch.setattr(
        "gemseo.optimization.multi_start.multi_start.execute", fake_execute
    )
    algo.execute(
        problem,
        settings=MultiStart_Settings(
            max_iter=10,
            doe_algo_settings=CustomDOE_Settings(
                samples=array([[0.2, 0.5, 0.4], [0.3, 0.8, 0.2]])
            ),
            skip_failed_starting_points=True,
            n_processes=2,
        ),
    )
    assert "skipping the starting point 1" in caplog.text
    assert "no sub-optimization problem" in caplog.text
    x_history = problem.database.get_x_vect_history()
    # There is no partial history to merge for the starting point
    # whose worker returned None.
    assert not any(allclose(x, [0.2, 0.5, 0.4]) for x in x_history)
    assert any(allclose(x, [0.3, 0.8, 0.2]) for x in x_history)
    assert problem.solution.x_opt is not None


class _FailingConstraintPower2(Power2):
    """A `Power2` problem whose first inequality constraint fails below a threshold.

    Unlike `_FailingPower2`, which fails inside the objective before it is ever
    evaluated, this variant fails after the objective has already been
    evaluated and stored for the current iterate, so that the sub-optimization
    database holds the objective but is missing this constraint.
    """

    def __init__(self, mode: str = "exception", threshold: float = 0.25) -> None:
        """
        Args:
            mode: The way the first inequality constraint fails,
                either by raising a `ValueError` (`"exception"`)
                or by returning a NaN value (`"nan"`).
            threshold: The value of `x[0]` below which the first inequality
                constraint fails.
        """  # noqa: D205 D212
        self._mode = mode
        self._threshold = threshold
        super().__init__()

    def ineq_constraint1(self, x_dv: RealArray) -> RealArray:
        """Compute the first inequality constraint, failing below the threshold.

        Args:
            x_dv: The design variable vector.

        Returns:
            The value of the first inequality constraint,
            or a NaN value in the `"nan"` failure mode.

        Raises:
            ValueError: In the `"exception"` failure mode,
                when `x[0]` is below the threshold.
        """
        if x_dv[0] < self._threshold:
            if self._mode == "exception":
                msg = "The discipline diverged."
                raise ValueError(msg)
            return array([nan])
        return super().ineq_constraint1(x_dv)


def test_partial_history_missing_constraint_is_merged(caplog):
    """Check that a design point missing a constraint value is merged.

    The first starting point's sub-optimization evaluates the objective at
    its very first iterate and then crashes while evaluating `ineq1`, so its
    database holds `pow2` at that design point but none of `ineq1`, `ineq2`
    or `eq`. This partial entry is still merged into the parent database,
    filtering it out would only hide the inconsistency rather than fix it;
    `OptimizationHistory.check_design_point_is_feasible` is responsible for
    treating a design point missing a constraint value as infeasible.
    """
    problem = _FailingConstraintPower2()
    algo = MultiStart()
    algo.execute(
        problem,
        settings=MultiStart_Settings(
            max_iter=10,
            doe_algo_settings=CustomDOE_Settings(
                samples=array([[0.1, 0.5, 0.4], [0.5, 0.5, 0.9]])
            ),
            skip_failed_starting_points=True,
            n_processes=1,
        ),
    )
    assert "skipping the starting point 1" in caplog.text
    x_history = problem.database.get_x_vect_history()
    # The failed seed's only design point is incomplete
    # (it is missing every constraint), but it is still merged.
    assert any(allclose(x, [0.1, 0.5, 0.4]) for x in x_history)
    # The second, successful seed still converges normally.
    assert any(allclose(x, [0.5, 0.5, 0.9]) for x in x_history)
    assert problem.solution.x_opt is not None
    # The merged, incomplete entry is indeed missing every constraint.
    outputs = next(
        outputs
        for x, outputs in zip(x_history, problem.database.values(), strict=True)
        if allclose(x, [0.1, 0.5, 0.4])
    )
    assert "ineq1" not in outputs
    assert "ineq2" not in outputs
    assert "eq" not in outputs


def test_partial_history_missing_constraint_is_merged_without_skipping():
    """Check that an incomplete design point is merged without skipping.

    Same invariant as
    [test_partial_history_missing_constraint_is_merged][test_partial_history_missing_constraint_is_merged],
    but with `skip_failed_starting_points` set to `False`. The first
    starting point's sub-optimization evaluates the objective at its first
    iterate and then gets a NaN value of `ineq1`, which the driver turns
    into a termination criterion rather than an exception, so `_optimize`
    returns a sub-problem whose database holds `pow2` at that design point
    but no value for `ineq2` or `eq`. This partial entry is still merged,
    since `OptimizationHistory.check_design_point_is_feasible` is
    responsible for treating it as infeasible, not the merge itself.
    """
    problem = _FailingConstraintPower2(mode="nan")
    algo = MultiStart()
    algo.execute(
        problem,
        settings=MultiStart_Settings(
            max_iter=10,
            opt_algo_settings=SLSQP_Settings(max_iter=3),
            doe_algo_settings=CustomDOE_Settings(
                samples=array([[0.1, 0.5, 0.4], [0.5, 0.5, 0.9]])
            ),
            skip_failed_starting_points=False,
            n_processes=1,
        ),
    )
    x_history = problem.database.get_x_vect_history()
    # The failed seed's incomplete point is merged, not dropped.
    assert any(allclose(x, [0.1, 0.5, 0.4]) for x in x_history)
    # The reported solution still comes from the surviving seed.
    assert problem.solution.x_opt is not None
    assert not allclose(problem.solution.x_opt, [0.1, 0.5, 0.4])


def test_partial_history_does_not_win_over_a_complete_point(caplog):
    """Check that a partial, objective-only point never wins the reported solution.

    The first starting point, `[0.1, 0.5, 0.4]`, fails as soon as `ineq1` is
    evaluated: its sub-optimization database only ever holds the objective at
    this design point, and no constraint value at all. This incomplete entry
    is merged into the parent database like any other, with the objective
    but no constraint values. Thanks to
    [OptimizationHistory.check_design_point_is_feasible][gemseo.optimization.history.OptimizationHistory.check_design_point_is_feasible]
    treating a design point missing a constraint value as infeasible with an
    infinite violation measure, `__get_best_infeasible_point` still prefers
    the fully evaluated, genuinely infeasible points of the surviving seed,
    `[0.3, 0.3, 0.3]`, which does not reach feasibility within its
    (deliberately small) iteration budget.
    """
    problem = _FailingConstraintPower2()
    algo = MultiStart()
    algo.execute(
        problem,
        settings=MultiStart_Settings(
            max_iter=10,
            opt_algo_settings=SLSQP_Settings(max_iter=2),
            doe_algo_settings=CustomDOE_Settings(
                samples=array([[0.1, 0.5, 0.4], [0.3, 0.3, 0.3]])
            ),
            skip_failed_starting_points=True,
            n_processes=1,
        ),
    )
    assert "skipping the starting point 1" in caplog.text
    # The reported solution must come from the surviving seed,
    # not from the failed seed's partial, objective-only point.
    assert problem.solution.x_opt is not None
    assert not allclose(problem.solution.x_opt, [0.1, 0.5, 0.4])
    assert problem.solution.x_opt[0] >= 0.25
    # The failed seed's incomplete point is still merged into the database.
    x_history = problem.database.get_x_vect_history()
    assert any(allclose(x, [0.1, 0.5, 0.4]) for x in x_history)


@pytest.mark.parametrize("n_processes", [1, 2])
def test_all_starting_points_fail(n_processes, snapshot):
    """Check that a `ValueError` is raised when every starting point fails."""
    problem = _FailingPower2(mode="exception", threshold=0.9)
    algo = MultiStart()
    with assert_exception(ValueError, snapshot):
        algo.execute(
            problem,
            settings=MultiStart_Settings(
                max_iter=10,
                doe_algo_settings=CustomDOE_Settings(
                    samples=array([[0.2, 0.5, 0.4], [0.3, 0.8, 0.2]])
                ),
                skip_failed_starting_points=True,
                n_processes=n_processes,
            ),
        )


# The 1-based index of the objective call that fails in the two tests below.
# The multi-start algorithm evaluates the objective once at the current
# design value before the sub-optimizations (call 1); the first
# starting point's sub-optimization then evaluates it twice more
# (calls 2 and 3) before the crash on call 4, leaving the second starting
# point's sub-optimization (calls 5 onward) unaffected.
_FAILING_CALL_INDEX = 4


def test_partial_history_is_merged(caplog):
    """Check that the pre-crash points of a failed seed are kept.

    The first starting point, `[0.85, 0.85, 0.9655]`,
    fails on the fourth objective call,
    after its sub-optimization has already evaluated it twice;
    the second starting point, `[0.99, 0.99, 0.9655]`,
    is evaluated afterward and its sub-optimization completes normally.
    """
    problem = _FailingPower2(mode="exception", failing_call_index=_FAILING_CALL_INDEX)
    algo = MultiStart()
    algo.execute(
        problem,
        settings=MultiStart_Settings(
            max_iter=10,
            opt_algo_settings=SLSQP_Settings(max_iter=3),
            doe_algo_settings=CustomDOE_Settings(
                samples=array([[0.85, 0.85, 0.9655], [0.99, 0.99, 0.9655]])
            ),
            skip_failed_starting_points=True,
            n_processes=1,
        ),
    )
    # The first starting point is classified as failed, not silently successful.
    assert "skipping the starting point 1" in caplog.text

    x_history = problem.database.get_x_vect_history()

    # The initial evaluation (call 1) and the failed seed's starting point
    # and first iterate (calls 2 and 3) all succeeded before the crash.
    assert len(problem.evaluated_x) > _FAILING_CALL_INDEX - 1
    failed_seed_evaluated_x = problem.evaluated_x[1 : _FAILING_CALL_INDEX - 1]
    assert len(failed_seed_evaluated_x) == 2
    # The failed seed's pre-crash points reached the merged database.
    for x in failed_seed_evaluated_x:
        assert any(allclose(x, xi) for xi in x_history)

    # The failing design value never reaches the merged database.
    assert problem.failing_x is not None
    assert not any(allclose(problem.failing_x, x) for x in x_history)

    assert problem.solution.x_opt is not None


def test_partial_history_is_merged_with_multistart_file_path(tmp_wd):
    """Check that the partial history of a failed seed does not reach the HDF file.

    Same arrangement as
    [test_partial_history_is_merged][test_partial_history_is_merged]:
    the pre-crash points of the failed seed must still be merged into the
    parent database, while the HDF file must only hold the surviving seed's
    local optimum, and neither the failed seed's starting point nor its
    failing design value.
    """
    problem = _FailingPower2(mode="exception", failing_call_index=_FAILING_CALL_INDEX)
    algo = MultiStart()
    algo.execute(
        problem,
        settings=MultiStart_Settings(
            max_iter=10,
            opt_algo_settings=SLSQP_Settings(max_iter=3),
            doe_algo_settings=CustomDOE_Settings(
                samples=array([[0.85, 0.85, 0.9655], [0.99, 0.99, 0.9655]])
            ),
            skip_failed_starting_points=True,
            n_processes=1,
            multistart_file_path="local_optima.hdf5",
        ),
    )
    x_history = problem.database.get_x_vect_history()
    # The parent database keeps the pre-crash points of the failed seed...
    pre_crash_evaluated_x = problem.evaluated_x[: _FAILING_CALL_INDEX - 1]
    failed_seed_evaluated_x = pre_crash_evaluated_x[1:]
    for x in failed_seed_evaluated_x:
        assert any(allclose(x, xi) for xi in x_history)
    # ... as well as the surviving seed's points.
    assert any(allclose(x, [0.99, 0.99, 0.9655]) for x in x_history)

    # But the HDF file only holds the surviving seed's local optimum.
    hdf_database = Power2.from_hdf("local_optima.hdf5").database
    assert len(hdf_database) == 1
    (x_opt,) = hdf_database.get_x_vect_history()
    # It is neither the failed seed's starting point nor its failing design
    # value, and it lies in the neighbourhood of the surviving seed,
    # i.e. it was evaluated after the crash.
    assert not any(allclose(x_opt, x) for x in failed_seed_evaluated_x)
    assert not allclose(x_opt, problem.failing_x)
    assert any(
        allclose(x_opt, x) for x in problem.evaluated_x[_FAILING_CALL_INDEX - 1 :]
    )


@pytest.mark.parametrize("skip_failed_starting_points", [True, False])
def test_full_history_is_merged_with_observable(skip_failed_starting_points):
    """Check that a full history is merged when the problem has an observable.

    `_optimize` only copies the parent problem's observables into the
    sub-problem's `observables` collection, never into its
    `new_iter_observables`, so an observable is never evaluated during a
    sub-optimization; every sub-problem database entry is thus missing a
    value for `obs`. The merge no longer filters entries by completeness, so
    every entry of every seed still reaches the parent database, even though
    none of them ever holds the observable on its own.
    """
    problem = Power2()
    problem.add_observable(
        ArrayFunction(operator.itemgetter(0), name="obs", input_names=["x"])
    )
    algo = MultiStart()
    algo.execute(
        problem,
        settings=MultiStart_Settings(
            max_iter=30,
            doe_algo_settings=CustomDOE_Settings(
                samples=array([[0.2, 0.5, 0.4], [0.3, 0.8, 0.2], [0.6, 0.4, 0.7]])
            ),
            skip_failed_starting_points=skip_failed_starting_points,
            n_processes=1,
        ),
    )
    # Every seed converges normally, so the merged database must hold far
    # more than one entry per seed, not just a single (initial-point) entry.
    assert len(problem.database) > 9
    # The observable is never evaluated by a sub-optimization; the parent
    # problem's new-iteration listener evaluates it when the merge stores
    # the design value, so every merged entry ends up holding it.
    assert all("obs" in outputs for outputs in problem.database.values())
    # The reported optimum is the real optimum of the problem, not the
    # (infeasible) initial design value, whose objective value is 3.
    assert problem.solution.f_opt < 3.0
    assert_almost_equal(
        problem.solution.x_opt,
        array([0.5 ** (1.0 / 3.0), 0.5 ** (1.0 / 3.0), 0.9 ** (1.0 / 3.0)]),
        decimal=3,
    )


def test_jacobians_are_merged():
    """Check that the Jacobians evaluated by the sub-optimizations are merged.

    The default sub-optimization algorithm, SLSQP, is gradient-based and
    stores the Jacobians it evaluates, so the parent database must hold the
    gradient-tagged names of the objective and of every constraint.
    """
    problem = Power2()
    algo = MultiStart()
    algo.execute(
        problem,
        settings=MultiStart_Settings(
            max_iter=10,
            doe_algo_settings=CustomDOE_Settings(
                samples=array([[0.2, 0.5, 0.4], [0.3, 0.8, 0.2]])
            ),
        ),
    )
    function_names = problem.database.get_function_names(skip_grad=False)
    for name in (problem.objective.name, *problem.constraints.get_names()):
        assert Database.get_gradient_name(name) in function_names


def test_jacobians_are_not_merged_when_store_jacobian_is_false():
    """Check that the Jacobians are not merged when store_jacobian is disabled.

    The default sub-optimization algorithm, SLSQP, is gradient-based and
    stores the Jacobians it evaluates, but with `store_jacobian=False` the
    parent database must not hold the gradient-tagged names of the objective
    or of any constraint, while the plain function names are still merged.
    """
    problem = Power2()
    algo = MultiStart()
    algo.execute(
        problem,
        settings=MultiStart_Settings(
            max_iter=10,
            doe_algo_settings=CustomDOE_Settings(
                samples=array([[0.2, 0.5, 0.4], [0.3, 0.8, 0.2]])
            ),
            store_jacobian=False,
        ),
    )
    function_names = problem.database.get_function_names(skip_grad=False)
    for name in (problem.objective.name, *problem.constraints.get_names()):
        assert name in function_names
        assert Database.get_gradient_name(name) not in function_names


@pytest.mark.parametrize("store_jacobian", [True, False])
def test_multistart_file_path_honors_store_jacobian(tmp_wd, store_jacobian):
    """Check that the file written by multistart_file_path honors store_jacobian.

    The default sub-optimization algorithm, SLSQP, is gradient-based and
    stores the Jacobians it evaluates. The HDF file written by
    `multistart_file_path` must hold the gradient-tagged names of the
    objective and of every constraint when `store_jacobian` is `True` (the
    default), and none of them when it is `False`, while the plain function
    names are stored either way.
    """
    problem = Power2()
    algo = MultiStart()
    algo.execute(
        problem,
        settings=MultiStart_Settings(
            max_iter=10,
            doe_algo_settings=CustomDOE_Settings(
                samples=array([[0.2, 0.5, 0.4], [0.3, 0.8, 0.2]])
            ),
            store_jacobian=store_jacobian,
            multistart_file_path="local_optima.hdf5",
        ),
    )
    hdf_database = Power2.from_hdf("local_optima.hdf5").database
    function_names = hdf_database.get_function_names(skip_grad=False)
    for name in (problem.objective.name, *problem.constraints.get_names()):
        assert name in function_names
        if store_jacobian:
            assert Database.get_gradient_name(name) in function_names
        else:
            assert Database.get_gradient_name(name) not in function_names


def test_multistart_file_path_with_failed_start(tmp_wd):
    """Check that multistart_file_path only stores the surviving seeds."""
    problem = _FailingPower2(mode="exception")
    algo = MultiStart()
    algo.execute(
        problem,
        settings=MultiStart_Settings(
            max_iter=15,
            doe_algo_settings=CustomDOE_Settings(
                samples=array([[0.1, 0.5, 0.4], [0.5, 0.5, 0.9], [0.6, 0.7, 0.5]])
            ),
            skip_failed_starting_points=True,
            multistart_file_path="local_optima.hdf5",
        ),
    )
    assert len(Power2.from_hdf("local_optima.hdf5").database) == 2


def test_multistart_file_path_stores_incomplete_local_optimum(tmp_wd):
    """Check that multistart_file_path stores an incomplete local optimum too.

    Same arrangement as
    [test_partial_history_missing_constraint_is_merged_without_skipping][test_partial_history_missing_constraint_is_merged_without_skipping]:
    the first starting point's sub-optimization evaluates the objective at
    its first iterate and then gets a NaN value of `ineq1`, so `_optimize`
    returns a sub-problem whose reported `x_opt` is the incomplete first
    iterate, missing a constraint value. This design value is still written
    to the HDF file, like any other local optimum; filtering it out would
    only make the file inconsistent with the parent database. The surviving
    seed is given enough iterations to actually converge to the analytic
    optimum.
    """
    problem = _FailingConstraintPower2(mode="nan")
    algo = MultiStart()
    algo.execute(
        problem,
        settings=MultiStart_Settings(
            max_iter=100,
            opt_algo_settings=SLSQP_Settings(max_iter=20),
            doe_algo_settings=CustomDOE_Settings(
                samples=array([[0.1, 0.5, 0.4], [0.5, 0.5, 0.9]])
            ),
            skip_failed_starting_points=False,
            n_processes=1,
            multistart_file_path="local_optima.hdf5",
        ),
    )
    hdf_database = Power2.from_hdf("local_optima.hdf5").database
    # Both the failed seed's incomplete local optimum
    # and the surviving seed's local optimum are stored.
    assert len(hdf_database) == 2
    x_vects = hdf_database.get_x_vect_history()
    assert any(
        allclose(
            x,
            array([0.5 ** (1.0 / 3.0), 0.5 ** (1.0 / 3.0), 0.9 ** (1.0 / 3.0)]),
            atol=1e-3,
        )
        for x in x_vects
    )


def _fake_sub_optimization(skipped_x: RealArray | None = None):
    """Return a fake `execute` evaluating nothing from some starting points.

    Args:
        skipped_x: The starting point from which the fake sub-optimization
            evaluates nothing.
            If `None`, it evaluates nothing from any starting point.

    Returns:
        A replacement of `optimization_library_factory.execute`
        delegating to the real one for the other starting points.
    """
    real_execute = optimization_library_factory.execute

    def fake_execute(problem, settings):
        """Run the sub-optimization, or do nothing from the skipped starting point.

        Args:
            problem: The sub-optimization problem.
            settings: The settings of the sub-optimization algorithm.
        """
        if skipped_x is None or allclose(
            problem.input_space.get_current_value(), skipped_x
        ):
            return None
        return real_execute(problem, settings=settings)

    return fake_execute


def test_skip_starting_point_without_objective_evaluation(monkeypatch, caplog):
    """Check that a sub-optimization evaluating no objective value is skipped."""
    monkeypatch.setattr(
        optimization_library_factory,
        "execute",
        _fake_sub_optimization(array([0.2, 0.5, 0.4])),
    )
    problem = Power2()
    MultiStart().execute(
        problem,
        settings=MultiStart_Settings(
            max_iter=10,
            doe_algo_settings=CustomDOE_Settings(
                samples=array([[0.2, 0.5, 0.4], [0.3, 0.8, 0.2]])
            ),
        ),
    )
    assert "skipping the starting point 1" in caplog.text
    assert "no evaluation of the objective 'pow2'" in caplog.text
    assert any(
        allclose(x, [0.3, 0.8, 0.2]) for x in problem.database.get_x_vect_history()
    )


def test_no_objective_evaluation_without_skipping(monkeypatch, snapshot):
    """Check the error raised when a sub-optimization evaluates no objective value.

    This error is raised only when `skip_failed_starting_points` is `False`.
    """
    monkeypatch.setattr(
        optimization_library_factory, "execute", _fake_sub_optimization()
    )
    with assert_exception(ValueError, snapshot):
        MultiStart().execute(
            Power2(),
            settings=MultiStart_Settings(
                max_iter=10,
                doe_algo_settings=CustomDOE_Settings(
                    samples=array([[0.2, 0.5, 0.4], [0.3, 0.8, 0.2]])
                ),
                skip_failed_starting_points=False,
            ),
        )
