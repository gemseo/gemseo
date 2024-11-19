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

from copy import copy
from copy import deepcopy
from timeit import default_timer as timer

import pytest
from numpy import array
from numpy import complex128
from numpy import equal
from numpy import ones
from scipy.optimize import rosen

from gemseo import create_design_space
from gemseo import create_discipline
from gemseo import create_scenario
from gemseo.core.mdo_functions.discipline_adapter_generator import (
    DisciplineAdapterGenerator,
)
from gemseo.core.parallel_execution.callable_parallel_execution import (
    CallableParallelExecution,
)
from gemseo.core.parallel_execution.disc_parallel_execution import DiscParallelExecution
from gemseo.core.parallel_execution.disc_parallel_linearization import (
    DiscParallelLinearization,
)
from gemseo.problems.mdo.sellar.sellar_1 import Sellar1
from gemseo.problems.mdo.sellar.sellar_2 import Sellar2
from gemseo.problems.mdo.sellar.sellar_system import SellarSystem
from gemseo.problems.mdo.sellar.utils import WITH_2D_ARRAY
from gemseo.problems.mdo.sellar.utils import get_initial_data
from gemseo.problems.mdo.sellar.variables import X_SHARED
from gemseo.problems.mdo.sellar.variables import Y_1
from gemseo.utils.platform import PLATFORM_IS_WINDOWS


class CallableWorker:
    """Callable worker."""

    def __call__(self, counter):
        """Callable."""
        return 2 * counter


def function_raising_exception(counter) -> None:
    """Raises an Exception."""
    msg = "This is an Exception"
    raise RuntimeError(msg)


def test_functional() -> None:
    """Test the execution of functions in parallel."""
    n = 10
    parallel_execution = CallableParallelExecution([rosen])
    output_list = parallel_execution.execute([[0.5] * i for i in range(1, n + 1)])
    assert output_list == [rosen([0.5] * i) for i in range(1, n + 1)]


@pytest.mark.parametrize("as_iterable", [False, True])
def test_callback(as_iterable) -> None:
    """Test the execution of callbacks."""

    class Counter:
        def __init__(self):
            self.total = 0

        def callback(self, index, data):
            self.total += 2 * index + (data == 0)

    counter = Counter()
    callback = counter.callback
    exec_callback = [callback] if as_iterable else callback
    parallel_execution = CallableParallelExecution([rosen])
    output_data = parallel_execution.execute(
        [[1.0, 1.0], [1.0, 1.0, 1.0]], exec_callback=exec_callback
    )
    assert output_data == [0.0, 0.0]
    assert counter.total == 4.0


def test_callback_error() -> None:
    parallel_execution = CallableParallelExecution([rosen])
    with pytest.raises(TypeError):
        parallel_execution.execute(
            [[0.5] * i for i in range(1, 3)], exec_callback="toto"
        )


def test_task_submitted_callback_error() -> None:
    function_list = [rosen] * 3
    parallel_execution = CallableParallelExecution(function_list)

    with pytest.raises(TypeError):
        parallel_execution.execute(
            [[0.5] * i for i in range(1, 3)],
            task_submitted_callback="not_callable",
        )


def test_callable() -> None:
    """Test CallableParallelExecution with a Callable worker."""
    n = 2
    function_list = [CallableWorker(), CallableWorker()]
    parallel_execution = CallableParallelExecution(function_list, use_threading=True)
    output_list = parallel_execution.execute([1] * n)
    assert output_list == [2] * n


def test_callable_exception() -> None:
    """Test CallableParallelExecution with a Callable worker."""
    n = 2
    function_list = [function_raising_exception, CallableWorker()]
    parallel_execution = CallableParallelExecution(function_list, use_threading=True)
    parallel_execution.execute([1] * n)


def test_disc_parallel_doe_scenario() -> None:
    s_1 = Sellar1()
    design_space = create_design_space()
    design_space.add_variable("x_1", lower_bound=0.0, value=1.0, upper_bound=10.0)
    scenario = create_scenario(
        s_1, Y_1, design_space, scenario_type="DOE", formulation_name="DisciplinaryOpt"
    )
    n_samples = 20
    scenario.execute(
        algo_name="LHS",
        n_samples=n_samples,
        eval_jac=True,
        n_processes=2,
    )
    assert (
        len(
            scenario.formulation.optimization_problem.database.get_function_history(Y_1)
        )
        == n_samples
    )


def test_disc_parallel_doe(sellar_disciplines) -> None:
    """Test the execution of disciplines in parallel."""
    s_1 = sellar_disciplines.sellar1
    n = 10
    parallel_execution = DiscParallelExecution(
        [s_1], n_processes=2, wait_time_between_fork=0.1
    )
    input_list = []
    for i in range(n):
        inputs = get_initial_data()
        if WITH_2D_ARRAY:  # pragma: no cover
            inputs[X_SHARED][0][0] = i
        else:
            inputs[X_SHARED][0] = i
        input_list.append(inputs)

    t_0 = timer()
    outs = parallel_execution.execute(input_list)
    t_f = timer()

    elapsed_time = t_f - t_0
    assert elapsed_time > 0.1 * (n - 1)

    assert s_1.execution_statistics.n_executions == n

    func_gen = DisciplineAdapterGenerator(s_1)
    y_0_func = func_gen.get_function([X_SHARED], [Y_1])

    parallel_execution = CallableParallelExecution([y_0_func.evaluate])
    input_list = [array([i, 0], dtype=complex128) for i in range(n)]
    output_list = parallel_execution.execute(input_list)

    for i in range(n):
        inputs = get_initial_data()
        if WITH_2D_ARRAY:
            inputs[X_SHARED][0][0] = i
        else:
            inputs[X_SHARED][0] = i
        s_1.execute(inputs)
        assert s_1.io.data[Y_1] == outs[i][Y_1]
        assert s_1.io.data[Y_1] == output_list[i]


def test_parallel_lin() -> None:
    disciplines = [Sellar1(), Sellar2(), SellarSystem()]
    parallel_execution = DiscParallelLinearization(disciplines)

    input_list = []
    for i in range(3):
        inpts = get_initial_data()
        inpts[X_SHARED][0] = i + 1
        input_list.append(inpts)
    outs = parallel_execution.execute(input_list)

    disciplines2 = [Sellar1(), Sellar2(), SellarSystem()]

    for i, disc in enumerate(disciplines):
        inpts = get_initial_data()
        inpts[X_SHARED][0] = i + 1

        j_ref = disciplines2[i].linearize(inpts)

        for f, jac_loc in disc.jac.items():
            for x, dfdx in jac_loc.items():
                assert (dfdx == j_ref[f][x]).all()
                assert (dfdx == outs[i][f][x]).all()


def test_disc_parallel_threading_proc(sellar_disciplines) -> None:
    disciplines = copy(sellar_disciplines)
    parallel_execution = DiscParallelExecution(
        disciplines, n_processes=2, use_threading=True
    )
    outs1 = parallel_execution.execute([None] * 3)

    disciplines = copy(sellar_disciplines)
    parallel_execution = DiscParallelExecution(disciplines, n_processes=2)
    outs2 = parallel_execution.execute([None] * 3)

    for out_d1, out_d2 in zip(outs1, outs2):
        for name, val in out_d2.items():
            assert equal(out_d1[name], val).all()

    disciplines = [sellar_disciplines.sellar1] * 2

    with pytest.raises(ValueError):
        DiscParallelExecution(
            disciplines,
            n_processes=2,
            use_threading=True,
        )


def test_async_call() -> None:
    disc = create_discipline("SobieskiMission")
    func = DisciplineAdapterGenerator(disc).get_function([X_SHARED], ["y_4"])

    x_list = [i * ones(6) for i in range(4)]

    def do_work():
        return list(map(func.evaluate, x_list))

    par = CallableParallelExecution([func.evaluate] * 2, n_processes=2)
    par.execute([i * ones(6) + 1 for i in range(2)], task_submitted_callback=do_work)


def test_not_worker(capfd) -> None:
    """Test that an exception is shown when a worker is not acceptable.

    The `TypeError` exception is caught by `worker`, but the execution continues.
    However, an error message has to be shown to the user.

    Args:
        capfd: Fixture capture outputs sent to `stdout` and
            `stderr`.
    """
    parallel_execution = CallableParallelExecution(["toto"])
    parallel_execution.execute([[0.5]])
    _, err = capfd.readouterr()
    assert err


def f(x: float = 0.0) -> float:
    """A function that raises an exception on certain conditions."""
    if x == 0:
        msg = "Undefined"
        raise ValueError(msg)
    return x + 1


@pytest.mark.parametrize(
    ("exceptions", "raises_exception"),
    [((), False), ((ValueError,), True), ((RuntimeError,), False)],
)
def test_re_raise_exceptions(exceptions, raises_exception) -> None:
    """Test that exceptions inside workers are properly handled.

    Args:
        exceptions: The exceptions that should not be ignored.
        raises_exception: Whether the input exception matches the one in the
            reference function.
    """
    parallel_execution = CallableParallelExecution(
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


@pytest.fixture
def reset_default_multiproc_method():
    """Reset the global multiprocessing method to the FORK method."""
    yield
    CallableParallelExecution.MULTI_PROCESSING_START_METHOD = (
        (CallableParallelExecution.MultiProcessingStartMethod.FORK)
        if not PLATFORM_IS_WINDOWS
        else CallableParallelExecution.MultiProcessingStartMethod.SPAWN
    )


@pytest.mark.parametrize(
    ("parallel_class", "n_calls_attr", "add_diff", "expected_n_calls"),
    [
        (DiscParallelExecution, "n_executions", False, 2),
        (DiscParallelLinearization, "n_linearizations", False, 0),
        (DiscParallelLinearization, "n_linearizations", True, 2),
    ],
)
@pytest.mark.parametrize(
    "mp_method",
    [
        (CallableParallelExecution.MultiProcessingStartMethod.FORK),
        (CallableParallelExecution.MultiProcessingStartMethod.SPAWN),
        ("threading"),
    ],
)
def test_multiprocessing_context(
    parallel_class,
    n_calls_attr,
    mp_method,
    add_diff,
    expected_n_calls,
    reset_default_multiproc_method,
) -> None:
    """Test the multiprocessing where the method for the context is changed.

    The test is applied on both parallel execution and linearization, with and without
    the definition of differentiated I/O.
    """

    # Just for the test purpose, we consider multithreading as a mp_method
    # and set the boolean ``use_threading`` from this.
    use_threading = mp_method == "threading"
    if not use_threading:
        CallableParallelExecution.MULTI_PROCESSING_START_METHOD = mp_method

    sellar = Sellar1()
    if add_diff:
        sellar.add_differentiated_inputs()
        sellar.add_differentiated_outputs()
    workers = [sellar, deepcopy(sellar)] if use_threading else [sellar]
    parallel_execution = parallel_class(workers, use_threading=use_threading)

    atom_inputs = get_initial_data()
    del atom_inputs[Y_1]
    atom_inputs_half = atom_inputs.copy()
    for name, value in atom_inputs_half.items():
        atom_inputs_half[name] = value / 2

    input_list = [atom_inputs, atom_inputs_half]

    if (
        PLATFORM_IS_WINDOWS
        and mp_method == CallableParallelExecution.MultiProcessingStartMethod.FORK
    ):
        with pytest.raises(ValueError):
            parallel_execution.execute(input_list)
    else:
        parallel_execution.execute(input_list)
        if use_threading:
            expected_n_calls /= 2
        assert getattr(sellar.execution_statistics, n_calls_attr) == expected_n_calls
