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
from typing import Any
from typing import Callable

from gemseo.core.parallel_execution.callable_parallel_execution import (
    CallableParallelExecution,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.algos.doe.doe_library import CallbackType
    from gemseo.typing import NumberArray


def execute(
    worker: Callable,
    callbacks: Iterable[CallbackType],
    n_processes: int,
    inputs: NumberArray | list[int],
) -> list[Any]:
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
        all_outputs = []
        for input_ in inputs:
            outputs = worker(input_)
            for callback in callbacks:
                callback(index=0, outputs=outputs)
            all_outputs.append(outputs)

        return all_outputs

    parallel_exec = CallableParallelExecution([worker], n_processes=n_processes)
    return parallel_exec.execute(inputs, exec_callback=callbacks)
