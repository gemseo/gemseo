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

import pytest
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.lib_custom import CustomDOE
from gemseo.algos.opt.opt_lib import OptimizationAlgorithmDescription
from gemseo.algos.opt.opt_lib import OptimizationLibrary
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.algos.opt_result import OptimizationResult
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from numpy import atleast_2d
from numpy.core._multiarray_umath import array
from numpy.core._multiarray_umath import zeros
from tqdm import tqdm


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


class ProgressOpt(OptimizationLibrary):
    OPTIONS_MAP = {}
    LIBRARY_NAME = "Test"

    def __init__(self, offsets, constraints_before_obj):
        super().__init__()
        self.descriptions = {
            "TestDriver": TestDesc(
                algorithm_name="TestDriver",
                description="d ",
                internal_algorithm_name="test",
                handle_equality_constraints=True,
                handle_inequality_constraints=True,
            ),
        }
        self.offsets = offsets
        self.constraints_before_obj = constraints_before_obj

    def _get_options(self, **options: Any) -> dict[str, Any]:
        return options

    def _run(self, **options: Any) -> OptimizationResult:
        """"""
        x_0, l_b, u_b = self.get_x0_and_bounds_vects(True)
        for off in self.offsets:
            if self.constraints_before_obj:
                self.problem.constraints[0].func(x_0 + off)
            self.problem.objective.func(x_0 + off)
        return self.get_optimum_from_database()


def test_progress_bar(
    caplog, offsets, constraints_before_obj, objective_and_problem_for_tests
):
    with caplog.at_level(logging.INFO):
        lib = ProgressOpt(offsets, constraints_before_obj)
        f, problem = objective_and_problem_for_tests
        lib.execute(problem, "TestDriver", max_iter=10)
        for k in range(len(offsets) + 1):
            assert f"{k * 10}%" in caplog.text
        count = zeros(len(offsets))
        for record in caplog.records:
            for k in range(len(offsets)):
                if f"{(k + 1) * 10}%" in record.message:
                    count[k] += 1
                    assert str(int(f(5.0 + offsets[k] * 10))) in record.message
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
        f_type=MDOFunction.TYPE_OBJ,
        expr="f(x)",
    )
    g = MDOFunction(
        func=dummy_sleep_function,
        name="g",
        f_type=MDOFunction.TYPE_INEQ,
        expr="g(x)",
    )
    design_space = DesignSpace()
    design_space.add_variable(
        "x", l_b=0.0, u_b=10.0, value=5.0, size=1, var_type=DesignSpace.FLOAT
    )
    problem = OptimizationProblem(design_space)
    problem.objective = f
    if constraints_before_obj:
        problem.add_constraint(g, 0.0, "ineq")
    return f, problem


def test_parallel_doe(caplog, offsets, objective_and_problem_for_tests):
    with caplog.at_level(logging.INFO):
        f, problem = objective_and_problem_for_tests
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
