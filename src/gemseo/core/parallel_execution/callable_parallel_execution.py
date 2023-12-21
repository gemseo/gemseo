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
#        :author: Charlie Vanaret, Francois Gallard, Gilberto Ruiz
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Parallel execution of disciplines and functions using multiprocessing."""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import sys
import threading as th
import time
import traceback
from multiprocessing import get_context
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Final
from typing import TypeVar

from docstring_inheritance import GoogleDocstringInheritanceMeta
from strenum import StrEnum

from gemseo.utils.multiprocessing import get_multi_processing_manager
from gemseo.utils.platform import PLATFORM_IS_WINDOWS

if TYPE_CHECKING:
    from collections.abc import Sequence

SUBPROCESS_NAME: Final[str] = "subprocess"

LOGGER = logging.getLogger(__name__)

_QueueType = TypeVar("_QueueType", queue.Queue, mp.Queue)


def _execute_workers(
    task_callables: _TaskCallables,
    queue_in: _QueueType,
    queue_out: _QueueType,
) -> None:
    """Call the task callables for args that are left in the queue_in.

    Args:
        task_callables: The task callables.
        queue_in: The queue with the task index to execute.
        queue_out: The queue object where the outputs of the workers are saved.
    """
    for task_index, input_ in iter(queue_in.get, None):
        try:
            sys.stdout.flush()
            output = task_callables(task_index, input_)
        except BaseException as err:
            traceback.print_exc()
            queue_out.put((task_index, err))
            queue_in.task_done()
            continue
        queue_out.put((task_index, output))
        queue_in.task_done()


class _TaskCallables:
    """Manage the call of one callable among callables."""

    callables: Sequence[Callable]
    """The callables."""

    inputs: Sequence[Any]
    """The inputs to be passed to the callables."""

    def __init__(self, callables: Sequence[Callable]) -> None:
        """
        Args:
            callables: The callables.
            inputs: The inputs to be passed to the callables.
        """  # noqa: D205, D212, D415
        self.callables = callables

    def __call__(self, task_index: int, input_) -> Any:
        """Call a callable.

        Args:
            task_index: The index of the callable to call.

        Returns:
            The output of callable.
        """
        if len(self.callables) > 1:
            callable_ = self.callables[task_index]
        else:
            callable_ = self.callables[0]
        return callable_(input_)


class CallableParallelExecution(metaclass=GoogleDocstringInheritanceMeta):
    """Perform a parallel execution of callables.

    The inputs must be independent objects.
    """

    class MultiProcessingStartMethod(StrEnum):
        """The multiprocessing start method."""

        FORK = "fork"
        SPAWN = "spawn"
        FORKSERVER = "forkserver"

    MULTI_PROCESSING_START_METHOD: ClassVar[MultiProcessingStartMethod]
    """The start method used for multiprocessing.

    The default is :attr:`.MultiProcessingStartMethod.SPAWN` on Windows,
    :attr:`.MultiProcessingStartMethod.FORK` otherwise.
    """

    if PLATFORM_IS_WINDOWS:
        MULTI_PROCESSING_START_METHOD = MultiProcessingStartMethod.SPAWN
    else:
        MULTI_PROCESSING_START_METHOD = MultiProcessingStartMethod.FORK

    N_CPUS: Final[int] = mp.cpu_count()
    """The number of CPUs."""

    workers: Sequence[Callable]
    """The objects that perform the tasks."""

    n_processes: int
    """The maximum simultaneous number of threads or processes."""

    use_threading: bool
    """Whether to use threads instead of processes to parallelize the execution."""

    wait_time_between_fork: float
    """The time to wait between two forks of the process/thread."""

    inputs: list[Any]
    """The inputs to be passed to the workers."""

    __exceptions_to_re_raise: tuple[type[Exception]]
    """The exception from a worker to be raised."""

    def __init__(
        self,
        workers: Sequence[Callable],
        n_processes: int = N_CPUS,
        use_threading: bool = False,
        wait_time_between_fork: float = 0.0,
        exceptions_to_re_raise: tuple[type[Exception]] = (),
    ) -> None:
        """
        Args:
            workers: The objects that perform the tasks.
                Either pass one worker, and it will be forked in multiprocessing.
                Or, when using multithreading or different workers, pass one worker
                per input data.
            n_processes: The maximum simultaneous number of threads,
                if ``use_threading`` is True, or processes otherwise,
                used to parallelize the execution.
            use_threading: Whether to use threads instead of processes
                to parallelize the execution.
                Multiprocessing will copy (serialize) all the disciplines,
                while threading will share all the memory.
                This is important to note if you want to execute the same
                discipline multiple times, in which case you shall use
                multiprocessing.
            wait_time_between_fork: The time to wait between two forks of the
                process/thread.
            exceptions_to_re_raise: The exceptions that should be raised again
                when caught inside a worker. If ``None``, all exceptions coming from
                workers are caught and the execution is allowed to continue.

        Raises:
            ValueError: If there are duplicated workers in ``workers`` when
                using multithreading.
        """  # noqa: D205, D212, D415
        self.workers = workers
        self.n_processes = n_processes
        self.use_threading = use_threading
        self.wait_time_between_fork = wait_time_between_fork
        self.__exceptions_to_re_raise = exceptions_to_re_raise
        self._check_unicity(workers)

    def _check_unicity(self, objects: Any) -> None:
        """Check that the objects are unique.

        Args:
            objects: The objects to check.
        """
        if self.use_threading:
            ids = {id(obj) for obj in objects}
            if len(ids) != len(objects):
                raise ValueError(
                    "When using multithreading, all workers shall be different objects."
                )

    def execute(
        self,
        inputs: Sequence[Any],
        exec_callback: Callable[[int, Any], Any] | None = None,
        task_submitted_callback: Callable | None = None,
    ) -> list[Any]:
        """Execute all the processes.

        Args:
            inputs: The input values.
            exec_callback: A callback called with the
                pair (index, outputs) as arguments when an item is retrieved
                from the processing. Index is the associated index
                in inputs of the input used to compute the outputs.
                If ``None``, no function is called.
            task_submitted_callback: A callback function called when all the
                tasks are submitted, but not done yet. If ``None``, no function
                is called.

        Returns:
            The computed outputs.

        Raises:
            TypeError: If the `exec_callback` is not callable.
                If the `task_submitted_callback` is not callable.

        Warnings:
            This class relies on multiprocessing features, it is therefore
            necessary to protect its execution with an ``if __name__ == '__main__':``
            statement when working on Windows.
        """
        if exec_callback is not None and not callable(exec_callback):
            raise TypeError("exec_callback function must be callable.")

        if task_submitted_callback is not None and not callable(
            task_submitted_callback
        ):
            raise TypeError("task_submitted_callback function must be callable.")

        n_tasks = len(inputs)

        tasks = list(range(n_tasks))[::-1]
        # Queue for workers.
        if self.use_threading:
            queue_in = queue.Queue()
            queue_out = queue.Queue()
            processor = th.Thread
        else:
            manager = get_multi_processing_manager()
            queue_in = manager.Queue()
            queue_out = manager.Queue()
            tasks = manager.list(tasks)
            self.__check_multiprocessing_start_method()
            processor = get_context(method=self.MULTI_PROCESSING_START_METHOD).Process

        task_callables = _TaskCallables(self.workers)

        processes = []
        for _ in range(min(n_tasks, self.n_processes)):
            proc = processor(
                target=_execute_workers,
                args=(task_callables, queue_in, queue_out),
                name=SUBPROCESS_NAME,
            )
            proc.daemon = True
            proc.start()
            processes.append(proc)

        if mp.current_process().name == SUBPROCESS_NAME and not self.use_threading:
            # The subprocesses do nothing here.
            return []

        # Fill the input queue.
        while tasks:
            task_index = tasks.pop()
            # Delay the next processes execution after the first one.
            if self.wait_time_between_fork > 0 and task_index > 0:
                time.sleep(self.wait_time_between_fork)
            queue_in.put((task_index, inputs[task_index]))

        if task_submitted_callback is not None:
            task_submitted_callback()

        # Sort the outputs with the same order as functions.
        ordered_outputs = [None] * n_tasks
        got_n_outs = 0
        # Retrieve outputs on the fly to call the callbacks, typically
        # iterates progress bar and stores the data in database or cache.
        stop = False

        while got_n_outs != n_tasks and not stop:
            index, output = queue_out.get()
            if isinstance(output, Exception):
                LOGGER.error("Failed to execute task indexed %s", str(index))
                LOGGER.error(output)
                # Condition to stop the execution only for required exceptions.
                # Otherwise, keep getting outputs from the queue.
                if isinstance(output, self.__exceptions_to_re_raise):
                    stop = True
            else:
                ordered_outputs[index] = output
                if exec_callback is not None:
                    exec_callback(index, output)
            got_n_outs += 1

        # Terminate the threads or processes.
        for _ in processes:
            queue_in.put(None)

        for proc in processes:
            proc.join()

        if isinstance(output, self.__exceptions_to_re_raise):
            raise output

        return ordered_outputs

    def __check_multiprocessing_start_method(self):
        """Check the multiprocessing start method with respect to the platform.

        Raises:
            ValueError: If the start method is different from ``spawn`` on
                Windows platform.
        """
        if (
            PLATFORM_IS_WINDOWS
            and self.MULTI_PROCESSING_START_METHOD
            != self.MultiProcessingStartMethod.SPAWN
        ):
            raise ValueError(
                f"The multiprocessing start method "
                f"{self.MULTI_PROCESSING_START_METHOD.value} "
                f"cannot be used on the Windows platform. "
                f"Only {self.MultiProcessingStartMethod.SPAWN.value} is available."
            )
