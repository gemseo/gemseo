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
from __future__ import annotations

import unittest
from timeit import default_timer as timer

import pytest
from gemseo.api import create_design_space
from gemseo.api import create_discipline
from gemseo.api import create_scenario
from gemseo.core.mdofunctions.function_generator import MDOFunctionGenerator
from gemseo.core.parallel_execution import DiscParallelExecution
from gemseo.core.parallel_execution import DiscParallelLinearization
from gemseo.core.parallel_execution import ParallelExecution
from gemseo.problems.sellar.sellar import get_inputs
from gemseo.problems.sellar.sellar import Sellar1
from gemseo.problems.sellar.sellar import Sellar2
from gemseo.problems.sellar.sellar import SellarSystem
from gemseo.problems.sellar.sellar import X_SHARED
from gemseo.problems.sellar.sellar import Y_1
from numpy import array
from numpy import complex128
from numpy import equal
from numpy import ones
from scipy.optimize import rosen


class CallableWorker:
    """Callable worker."""

    def __call__(self, counter):
        """Callable."""
        return 2 * counter


def function_raising_exception(counter):
    """Raises an Exception."""
    raise Exception("This is an Exception")


class TestParallelExecution(unittest.TestCase):
    """Test the parallel execution."""

    def test_functional(self):
        """Test the execution of functions in parallel."""
        n = 10
        function_list = rosen
        parallel_execution = ParallelExecution(function_list)
        output_list = parallel_execution.execute([[0.5] * i for i in range(1, n + 1)])
        assert output_list == [rosen([0.5] * i) for i in range(1, n + 1)]

        self.assertRaises(
            TypeError,
            parallel_execution.execute,
            [[0.5] * i for i in range(1, n + 1)],
            exec_callback="toto",
        )

        function_list = [rosen] * n
        parallel_execution = ParallelExecution(function_list)
        self.assertRaises(
            TypeError,
            parallel_execution.execute,
            [[0.5] * i for i in range(1, n + 1)],
            task_submitted_callback="not_callable",
        )

    def test_callable(self):
        """Test ParallelExecution with a Callable worker."""
        n = 2
        function_list = [CallableWorker(), CallableWorker()]
        parallel_execution = ParallelExecution(function_list, use_threading=True)
        output_list = parallel_execution.execute([1] * n)
        assert output_list == [2] * n

    def test_callable_exception(self):
        """Test ParallelExecution with a Callable worker."""
        n = 2
        function_list = [function_raising_exception, CallableWorker()]
        parallel_execution = ParallelExecution(function_list, use_threading=True)
        parallel_execution.execute([1] * n)

    def test_disc_parallel_doe_scenario(self):
        s_1 = Sellar1()
        design_space = create_design_space()
        design_space.add_variable("x_local", l_b=0.0, value=1.0, u_b=10.0)
        scenario = create_scenario(
            s_1, "DisciplinaryOpt", "y_1", design_space, scenario_type="DOE"
        )
        n_samples = 20
        scenario.execute(
            {
                "algo": "lhs",
                "n_samples": n_samples,
                "algo_options": {"eval_jac": True, "n_processes": 2},
            }
        )
        assert (
            len(scenario.formulation.opt_problem.database.get_func_history("y_1"))
            == n_samples
        )

    def test_disc_parallel_doe(self):
        """Test the execution of disciplines in parallel."""
        s_1 = Sellar1()
        n = 10
        parallel_execution = DiscParallelExecution(
            s_1, n_processes=2, wait_time_between_fork=0.1
        )
        input_list = []
        for i in range(n):
            inpts = get_inputs()
            inpts[X_SHARED][0] = i
            input_list.append(inpts)

        t_0 = timer()
        outs = parallel_execution.execute(input_list)
        t_f = timer()

        elapsed_time = t_f - t_0
        assert elapsed_time > 0.1 * (n - 1)

        assert s_1.n_calls == n

        func_gen = MDOFunctionGenerator(s_1)
        y_0_func = func_gen.get_function([X_SHARED], [Y_1])

        parallel_execution = ParallelExecution(y_0_func)
        input_list = [array([i, 0.0], dtype=complex128) for i in range(n)]
        output_list = parallel_execution.execute(input_list)

        for i in range(n):
            inpts = get_inputs()
            inpts[X_SHARED][0] = i
            s_1.execute(inpts)
            assert s_1.local_data[Y_1] == outs[i][Y_1]
            assert s_1.local_data[Y_1] == output_list[i]

    def test_parallel_lin(self):
        disciplines = [Sellar1(), Sellar2(), SellarSystem()]
        parallel_execution = DiscParallelLinearization(disciplines)

        input_list = []
        for i in range(3):
            inpts = get_inputs()
            inpts[X_SHARED][0] = i + 1
            input_list.append(inpts)
        outs = parallel_execution.execute(input_list)

        disciplines2 = [Sellar1(), Sellar2(), SellarSystem()]

        for i, disc in enumerate(disciplines):
            inpts = get_inputs()
            inpts[X_SHARED][0] = i + 1

            j_ref = disciplines2[i].linearize(inpts)

            for f, jac_loc in disc.jac.items():
                for x, dfdx in jac_loc.items():
                    assert (dfdx == j_ref[f][x]).all()
                    assert (dfdx == outs[i][f][x]).all()

    def test_disc_parallel_threading_proc(self):
        disciplines = [Sellar1(), Sellar2(), SellarSystem()]
        parallel_execution = DiscParallelExecution(
            disciplines, n_processes=2, use_threading=True
        )
        outs1 = parallel_execution.execute([None] * 3)

        disciplines = [Sellar1(), Sellar2(), SellarSystem()]
        parallel_execution = DiscParallelExecution(disciplines, n_processes=2)
        outs2 = parallel_execution.execute([None] * 3)

        for out_d1, out_d2 in zip(outs1, outs2):
            for name, val in out_d2.items():
                assert equal(out_d1[name], val).all()

        disciplines = [Sellar1()] * 2
        self.assertRaises(
            ValueError,
            DiscParallelExecution,
            disciplines,
            n_processes=2,
            use_threading=True,
        )

    def test_async_call(self):
        disc = create_discipline("SobieskiMission")
        func = MDOFunctionGenerator(disc).get_function([X_SHARED], ["y_4"])

        x_list = [i * ones(6) for i in range(4)]

        def do_work():
            return list(map(func, x_list))

        par = ParallelExecution([func] * 2, n_processes=2)
        par.execute(
            [i * ones(6) + 1 for i in range(2)], task_submitted_callback=do_work
        )


def test_not_worker(capfd):
    """Test that an exception is shown when a worker is not acceptable.

    The `TypeError` exception is caught by `worker`, but the execution continues.
    However, an error message has to be shown to the user.

    Args:
        capfd: Fixture capture outputs sent to `stdout` and
            `stderr`.
    """
    parallel_execution = ParallelExecution(["toto"])
    parallel_execution.execute([[0.5]])
    _, err = capfd.readouterr()
    assert err


def test_par_discipline_linearization():
    """Test the parallel linearization for a single worker."""
    sellar_par_lin = Sellar1()
    in_names = ["x_local", "x_shared", "y_2"]
    sellar_par_lin.add_differentiated_inputs(in_names)
    out_names = ["y_1"]
    sellar_par_lin.add_differentiated_outputs(out_names)

    parallel_execution = DiscParallelLinearization(sellar_par_lin)

    input_list = [
        {
            "x_local": array([0.0 + 0.0j]),
            "x_shared": array([1.0 + 0.0j, 0.0 + 0.0j]),
            "y_2": array([1.0 + 0.0j]),
        },
        {
            "x_local": array([0.5 + 0.0j]),
            "x_shared": array([0.5 + 0.0j, 0.0 + 0.0j]),
            "y_2": array([0.5 + 0.0j]),
        },
    ]

    parallel_execution.execute(input_list)

    assert sellar_par_lin.n_calls_linearize == 2


def test_par_discipline_lin_no_jac():
    """Test the parallel linearization for a single worker with no defined outputs."""
    sellar_par_lin = Sellar1()

    parallel_execution = DiscParallelLinearization(sellar_par_lin)

    input_list = [
        {
            "x_local": array([0.0 + 0.0j]),
            "x_shared": array([1.0 + 0.0j, 0.0 + 0.0j]),
            "y_2": array([1.0 + 0.0j]),
        },
        {
            "x_local": array([0.5 + 0.0j]),
            "x_shared": array([0.5 + 0.0j, 0.0 + 0.0j]),
            "y_2": array([0.5 + 0.0j]),
        },
    ]

    parallel_execution.execute(input_list)

    assert sellar_par_lin.n_calls_linearize == 0


def f(x: float = 0.0) -> float:
    """A function that raises an exception on certain conditions."""
    if x == 0:
        raise ValueError("Undefined")
    y = x + 1
    return y


@pytest.mark.parametrize(
    "exceptions,raises_exception",
    [(None, False), ((ValueError,), True), ((RuntimeError,), False)],
)
def test_re_raise_exceptions(exceptions, raises_exception):
    """Test that exceptions inside workers are properly handled.

    Args:
        exceptions: The exceptions that should not be ignored.
        raises_exception: Whether the input exception matches the one in the
            reference function.
    """
    parallel_execution = ParallelExecution(
        [f],
        n_processes=2,
        exceptions_to_re_raise=exceptions,
        wait_time_between_fork=0.1,
    )

    input_list = [array([1.0]), array([0.0])]

    if raises_exception:
        with pytest.raises(ValueError, match="Undefined"):
            parallel_execution.execute(input_list)
    else:
        assert parallel_execution.execute(input_list) == [array([2.0]), None]
