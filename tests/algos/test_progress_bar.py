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
from time import sleep
from typing import Any
from typing import ClassVar

import pytest
from numpy import atleast_2d
from numpy.core._multiarray_umath import array
from numpy.core._multiarray_umath import zeros
from tqdm import tqdm

from gemseo.algos._progress_bars.custom_tqdm_progress_bar import CustomTqdmProgressBar
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.custom_doe.custom_doe import CustomDOE
from gemseo.algos.opt.base_optimization_library import BaseOptimizationLibrary
from gemseo.algos.opt.base_optimization_library import OptimizationAlgorithmDescription
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.core.mdo_functions.mdo_function import MDOFunction


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

    def _run(self, problem: OptimizationProblem, **options: Any) -> None:
        x_0 = problem.design_space.get_current_value(
            complex_to_real=True, normalize=True
        )
        for off in self.offsets:
            if self.constraints_before_obj:
                problem.constraints[0].evaluate(x_0 + off)
            problem.objective.evaluate(x_0 + off)


def test_progress_bar(
    caplog, offsets, constraints_before_obj, objective_and_problem_for_tests
) -> None:
    with caplog.at_level(logging.INFO):
        lib = ProgressOpt(offsets, constraints_before_obj, "TestDriver")
        f, problem = objective_and_problem_for_tests
        lib.execute(problem, max_iter=10)
        for k in range(len(offsets) + 1):
            assert f"{k * 10}%" in caplog.text
        count = zeros(len(offsets))
        for record in caplog.records:
            for k in range(len(offsets)):
                if f"{(k + 1) * 10}%" in record.message:
                    count[k] += 1
                    assert str(int(f.evaluate(5.0 + offsets[k] * 10))) in record.message
                    if not constraints_before_obj and k >= 1:
                        assert tqdm.format_interval(0.1 * (k + 1)) in record.message
                        assert (
                            tqdm.format_interval(0.1 * (len(offsets) - (k + 1)))
                            in record.message
                        )
        assert max(count) == 1


@pytest.fixture
def objective_and_problem_for_tests(constraints_before_obj):
    f = MDOFunction(
        func=dummy_sleep_function,
        name="f",
        f_type=MDOFunction.FunctionType.OBJ,
        expr="f(x)",
    )
    g = MDOFunction(
        func=dummy_sleep_function,
        name="g",
        f_type=MDOFunction.ConstraintType.INEQ,
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
        problem.add_constraint(g, value=0.0, constraint_type="ineq")
    return f, problem


def test_parallel_doe(caplog, offsets, objective_and_problem_for_tests) -> None:
    with caplog.at_level(logging.INFO):
        _, problem = objective_and_problem_for_tests
        custom_doe = CustomDOE()

        i_k_0 = atleast_2d(array([offsets]) * 10 + 5).T
        custom_doe.execute(
            problem=problem,
            samples=i_k_0,
            n_processes=4,
        )
        for k in range(len(offsets) + 1):
            assert f"{k * 10}%" in caplog.text


def dummy_sleep_function(x):
    sleep(0.1)
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
