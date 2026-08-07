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
# Copyright 2022 IRT Saint Exupéry, https://www.irt-saintexupery.com
# Contributors:
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
from typing import ClassVar

import pytest
from numpy import array
from numpy import atleast_2d
from numpy import zeros
from tqdm import std as tqdm_std
from tqdm import tqdm

from gemseo.core.algorithm._progress_bar.custom import CustomTqdmProgressBar
from gemseo.core.algorithm.progress_bar_data.data import ProgressBarData
from gemseo.core.function.array_function import ArrayFunction
from gemseo.doe.custom_doe.custom_doe import CustomDOE
from gemseo.doe.custom_doe.settings.custom_doe_settings import CustomDOE_Settings
from gemseo.optimization.core.base_optimization_library import BaseOptimizationLibrary
from gemseo.optimization.core.base_optimization_library import (
    OptimizationAlgorithmDescription,
)
from gemseo.optimization.core.base_optimizer_settings import BaseOptimizerSettings
from gemseo.optimization.problem import OptimizationProblem
from gemseo.problem.optimization.rosenbrock import Rosenbrock
from gemseo.space.design import DesignSpace

TICK = 0.1
"""The duration in seconds of a call to the function to be evaluated."""


class Clock:
    """A clock advanced explicitly, to be used as the time source of tqdm.

    Unlike wall-clock time, this makes the elapsed and remaining times logged by the
    progress bar exactly predictable.
    """

    n_ticks: int
    """The number of ticks since the clock was reset."""

    def __init__(self) -> None:  # noqa: D107
        self.n_ticks = 0

    def tick(self) -> None:
        """Advance the clock by the duration of a function call."""
        self.n_ticks += 1

    def __call__(self) -> float:
        """Return the current time.

        Returns:
            The current time in seconds.
        """
        # Multiply instead of accumulating, to avoid the drift of the floating-point
        # additions; the expected times are computed with the very same expression.
        return TICK * self.n_ticks


CLOCK = Clock()


@pytest.fixture
def offsets():
    return [0.0, 0.3, 0.4, 0.5, 0.1, 0.2, -0.3, -0.1, -0.2, -0.4]


@pytest.fixture(params=[True, False])
def constraints_before_obj(request):
    return request.param


@dataclass
class TestDesc(OptimizationAlgorithmDescription):
    """Test driver."""

    library_name: str = "Test"


class TestDriver_Settings(BaseOptimizerSettings):  # noqa: N801
    """The settings of Test Driver."""


class ProgressOpt(BaseOptimizationLibrary):
    _OPTIONS_MAP: ClassVar[dict[Any, str]] = {}

    ALGORITHM_INFOS: ClassVar[dict[str, OptimizationAlgorithmDescription]] = {
        "TestDriver": TestDesc(
            algorithm_name="TestDriver",
            description="d ",
            internal_algorithm_name="test",
            handle_equality_constraints=True,
            handle_inequality_constraints=True,
        ),
    }

    def __init__(self, offsets, constraints_before_obj, algo_name) -> None:
        super().__init__(algo_name)
        self.offsets = offsets
        self.constraints_before_obj = constraints_before_obj

    def _get_options(self, **options: Any) -> dict[str, Any]:
        return options

    def _run(self, problem: OptimizationProblem) -> None:
        x_0 = problem.design_space.get_current_value(
            complex_to_real=True, normalize=True
        )
        for off in self.offsets:
            if self.constraints_before_obj:
                problem.constraints[0].evaluate(x_0 + off)
            problem.objective.evaluate(x_0 + off)


class NewProgressBarData(ProgressBarData):
    """The data of an optimization problem to be displayed in the progress bar."""


def test_progress_bar(
    caplog,
    monkeypatch,
    offsets,
    constraints_before_obj,
    objective_and_problem_for_tests,
) -> None:
    CLOCK.n_ticks = 0
    monkeypatch.setattr(tqdm_std, "time", CLOCK)
    with caplog.at_level(logging.INFO):
        lib = ProgressOpt(offsets, constraints_before_obj, "TestDriver")
        f, problem = objective_and_problem_for_tests
        lib.execute(problem, settings=TestDriver_Settings(max_iter=10))
        for k in range(len(offsets) + 1):
            assert f"{k * 10}%" in caplog.text
        # An iteration evaluates the objective, plus the constraint when there is one.
        n_calls_per_iteration = 2 if constraints_before_obj else 1
        count = zeros(len(offsets))
        for record in caplog.records:
            for k in range(len(offsets)):
                # Match the iteration counter rather than the percentage,
                # so that the progress bar of another test cannot be counted.
                if f"| {k + 1}/{len(offsets)} [" in record.message:
                    count[k] += 1
                    assert str(int(f.evaluate(5.0 + offsets[k] * 10))) in record.message
                    elapsed = tqdm.format_interval(
                        TICK * (n_calls_per_iteration * (k + 1))
                    )
                    remaining = tqdm.format_interval(
                        TICK * (n_calls_per_iteration * (len(offsets) - (k + 1)))
                    )
                    assert f"[{elapsed}<{remaining}," in record.message
        assert max(count) == 1


@pytest.fixture
def objective_and_problem_for_tests(constraints_before_obj):
    f = ArrayFunction(
        func=dummy_sleep_function,
        name="f",
        f_type=ArrayFunction.FunctionType.OBJ,
        expr="f(x)",
    )
    g = ArrayFunction(
        func=dummy_sleep_function,
        name="g",
        f_type=ArrayFunction.ConstraintType.INEQ,
        expr="g(x)",
    )
    design_space = DesignSpace()
    design_space.add_variable(
        "x",
        lower_bound=0.0,
        upper_bound=10.0,
        value=5.0,
        size=1,
        type_=DesignSpace.DesignVariableType.FLOAT,
    )
    problem = OptimizationProblem(design_space)
    problem.objective = f
    if constraints_before_obj:
        problem.add_constraint(
            g, value=0.0, constraint_type=problem.ConstraintType.INEQ
        )
    return f, problem


def test_parallel_doe(caplog, offsets, objective_and_problem_for_tests) -> None:
    with caplog.at_level(logging.INFO):
        _, problem = objective_and_problem_for_tests
        custom_doe = CustomDOE()

        i_k_0 = atleast_2d(array([offsets]) * 10 + 5).T
        custom_doe.execute(
            problem, settings=CustomDOE_Settings(samples=i_k_0, n_processes=4)
        )
        for k in range(len(offsets) + 1):
            assert f"{k * 10}%" in caplog.text


def dummy_sleep_function(x):
    CLOCK.tick()
    return -x


@pytest.mark.parametrize(
    ("e", "r"),
    [
        (1, " 1.00 it/sec"),
        (60 - 1, " 1.02 it/min"),
        (60, " 1.00 it/min"),
        (60 + 1, "59.02 it/hour"),
        (60 * 60 - 1, " 1.00 it/hour"),
        (60 * 60, " 1.00 it/hour"),
        (60 * 60 + 1, "23.99 it/day"),
        (60 * 60 * 24 - 1, " 1.00 it/day"),
        (60 * 60 * 24, " 1.00 it/day"),
        (60 * 60 * 24 + 1, " 1.00 it/day"),
    ],
)
def test_rate_expression(e, r) -> None:
    """Check CustomTqdmProgressBar.__get_rate_expression."""
    f = CustomTqdmProgressBar._CustomTqdmProgressBar__get_rate_expression
    assert f(1, e) == r


@pytest.mark.parametrize("n_processes", [1, 2])
def test_feasibility(caplog, n_processes):
    """Check that the feasibility is correctly logged."""
    problem = Rosenbrock()
    problem.add_constraint(
        ArrayFunction(sum, name="g"), value=0.0, constraint_type="ineq"
    )
    CustomDOE().execute(
        problem,
        settings=CustomDOE_Settings(
            samples=array([[-1.0, -1.0], [0.5, 0.5], [-0.5, -0.5]]),
            n_processes=n_processes,
        ),
    )
    assert "feas=True, obj=404" in caplog.text
    assert "feas=False, obj=6.5" in caplog.text
    assert "feas=True, obj=58.5" in caplog.text
