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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Multiprocessing execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.core.parallel_execution.callable_parallel_execution import ArgT
from gemseo.core.parallel_execution.callable_parallel_execution import (
    CallableParallelExecution,
)
from gemseo.core.parallel_execution.callable_parallel_execution import CallableType
from gemseo.core.parallel_execution.callable_parallel_execution import CallbackType
from gemseo.core.parallel_execution.callable_parallel_execution import ReturnT

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence


def execute(
    worker: CallableType[ArgT, ReturnT],
    callbacks: Iterable[CallbackType],
    n_processes: int,
    inputs: Sequence[ArgT],
) -> list[ReturnT | None]:
    """Run the worker with the given inputs in sequential or parallel mode.

    Args:
        worker: The object that performs the tasks.
        callbacks: Callback functions called with the
            pair (index, outputs) as arguments when an item is retrieved
            from the processing. Index is the associated index
            in inputs of the input used to compute the outputs.
            If empty, no function is called.
        n_processes: The number of processes used to evaluate the inputs.
        inputs: The inputs to be evaluated.

    Returns:
        The outputs of the evaluations.
    """
    if n_processes == 1:
        all_outputs: list[ReturnT | None] = []
        for input_ in inputs:
            outputs = worker(input_)
            for callback in callbacks:
                callback(0, outputs)
            all_outputs.append(outputs)

        return all_outputs

    parallel_exec = CallableParallelExecution([worker], n_processes=n_processes)
    return parallel_exec.execute(inputs, exec_callback=callbacks)
