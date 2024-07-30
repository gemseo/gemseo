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
import queue
import sys
import threading as th
import time
import traceback
from collections.abc import Callable
from multiprocessing import cpu_count
from multiprocessing import current_process
from multiprocessing import get_context
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Final
from typing import Generic
from typing import TypeVar
from typing import Union

from docstring_inheritance import GoogleDocstringInheritanceMeta
from strenum import StrEnum

from gemseo.utils.multiprocessing.manager import get_multi_processing_manager
from gemseo.utils.platform import PLATFORM_IS_WINDOWS

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence
    from multiprocessing.context import ForkProcess
    from multiprocessing.context import ForkServerProcess
    from multiprocessing.context import SpawnProcess
    from multiprocessing.managers import ListProxy

SUBPROCESS_NAME: Final[str] = "subprocess"

LOGGER = logging.getLogger(__name__)


CallbackType = Callable[[int, Any], None]
"""The type of a callback function."""

ArgT = TypeVar("ArgT")
ReturnT = TypeVar("ReturnT")

CallableType = Callable[[ArgT], ReturnT]

_QueueOutItem2 = Union[BaseException, ReturnT]

_QueueInType = queue.Queue[Union[None, tuple[int, ArgT]]]
_QueueOutType = queue.Queue[tuple[int, _QueueOutItem2[ReturnT]]]


def _execute_workers(
    task_callables: _TaskCallables[ArgT, ReturnT],
    queue_in: _QueueInType[ArgT],
    queue_out: _QueueOutType[ReturnT],
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
        except BaseException as err:  # noqa: BLE001
            traceback.print_exc()
            queue_out.put((task_index, err))
            queue_in.task_done()
            continue
        queue_out.put((task_index, output))
        queue_in.task_done()


class _TaskCallables(Generic[ArgT, ReturnT]):
    """Manage the call of one callable among callables."""

    callables: Sequence[CallableType[ArgT, ReturnT]]
    """The callables."""

    def __init__(self, callables: Sequence[CallableType[ArgT, ReturnT]]) -> None:
        """
        Args:
            callables: The callables.
        """  # noqa: D205, D212, D415
        self.callables = callables

    def __call__(self, task_index: int, input_: ArgT) -> ReturnT:
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


class CallableParallelExecution(
    Generic[ArgT, ReturnT], metaclass=GoogleDocstringInheritanceMeta
):
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

    if PLATFORM_IS_WINDOWS:  # pragma: win32 cover
        MULTI_PROCESSING_START_METHOD = MultiProcessingStartMethod.SPAWN
    else:  # pragma: win32 no cover
        MULTI_PROCESSING_START_METHOD = MultiProcessingStartMethod.FORK

    N_CPUS: Final[int] = cpu_count()
    """The number of CPUs."""

    workers: Sequence[CallableType[ArgT, ReturnT]]
    """The objects that perform the tasks."""

    n_processes: int
    """The maximum simultaneous number of threads or processes."""

    use_threading: bool
    """Whether to use threads instead of processes to parallelize the execution."""

    wait_time_between_fork: float
    """The time to wait between two forks of the process/thread."""

    inputs: list[Any]
    """The inputs to be passed to the workers."""

    __exceptions_to_re_raise: tuple[type[Exception], ...]
    """The exception from a worker to be raised."""

    def __init__(
        self,
        workers: Sequence[CallableType[ArgT, ReturnT]],
        n_processes: int = N_CPUS,
        use_threading: bool = False,
        wait_time_between_fork: float = 0.0,
        exceptions_to_re_raise: Sequence[type[Exception]] = (),
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
        self.__exceptions_to_re_raise = tuple(exceptions_to_re_raise)
        self._check_unicity(workers)

    def _check_unicity(self, objects: Any) -> None:
        """Check that the objects are unique.

        Args:
            objects: The objects to check.
        """
        if self.use_threading:
            ids = {id(obj) for obj in objects}
            if len(ids) != len(objects):
                msg = (
                    "When using multithreading, all workers shall be different objects."
                )
                raise ValueError(msg)

    # TODO: API: let exec_callback always be iterable and renamed to callbacks.
    def execute(
        self,
        inputs: Sequence[ArgT],
        exec_callback: CallbackType | Iterable[CallbackType] = (),
        task_submitted_callback: Callable[[], None] | None = None,
    ) -> list[ReturnT | None]:
        """Execute all the processes.

        Args:
            inputs: The input values.
            exec_callback: Callback functions called with the
                pair (index, outputs) as arguments when an item is retrieved
                from the processing. Index is the associated index
                in inputs of the input used to compute the outputs.
                If empty, no function is called.
            task_submitted_callback: A callback function called when all the
                tasks are submitted, but not done yet. If ``None``, no function
                is called.

        Returns:
            The computed outputs.

        Warnings:
            This class relies on multiprocessing features, it is therefore
            necessary to protect its execution with an ``if __name__ == '__main__':``
            statement when working on Windows.
        """
        if callable(exec_callback):
            exec_callback = [exec_callback]

        n_tasks = len(inputs)

        tasks: list[int] | ListProxy[int] = list(range(n_tasks))[::-1]

        queue_in: _QueueInType[ArgT]
        queue_out: _QueueOutType[ReturnT]
        processor: type[th.Thread | ForkProcess | SpawnProcess | ForkServerProcess]

        # TODO: API: use subclass instead of if?
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
            processor = get_context(method=self.MULTI_PROCESSING_START_METHOD).Process  # type: ignore[attr-defined]

        task_callables = _TaskCallables(self.workers)

        processes = []
        for _ in range(min(n_tasks, self.n_processes)):
            process = processor(
                target=_execute_workers,
                args=(task_callables, queue_in, queue_out),
                name=SUBPROCESS_NAME,
            )
            process.daemon = True
            process.start()
            processes.append(process)

        if current_process().name == SUBPROCESS_NAME and not self.use_threading:
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
        ordered_outputs: list[None | ReturnT] = [None] * n_tasks
        n_outputs = 0
        # Retrieve outputs on the fly to call the callbacks, typically
        # iterates progress bar and stores the data in database or cache.
        stop = False

        # TODO: simplify with for loop and build ordered_outputs incrementally.
        while n_outputs != n_tasks and not stop:
            index, output = queue_out.get()
            if isinstance(output, BaseException):
                LOGGER.error("Failed to execute task indexed %s", str(index))
                LOGGER.error(output)
                # Condition to stop the execution only for required exceptions.
                # Otherwise, keep getting outputs from the queue.
                if isinstance(output, self.__exceptions_to_re_raise):
                    stop = True
            else:
                ordered_outputs[index] = output
                for callback in exec_callback:
                    callback(index, output)
            n_outputs += 1

        # Terminate the threads or processes.
        for _ in processes:
            queue_in.put(None)

        for process in processes:
            process.join()

        if isinstance(output, self.__exceptions_to_re_raise):
            raise output

        return ordered_outputs

    def __check_multiprocessing_start_method(self) -> None:
        """Check the multiprocessing start method with respect to the platform.

        Raises:
            ValueError: If the start method is different from ``spawn`` on
                Windows platform.
        """
        if (
            PLATFORM_IS_WINDOWS
            and self.MULTI_PROCESSING_START_METHOD
            != self.MultiProcessingStartMethod.SPAWN
        ):  # pragma: win32 cover
            msg = (
                f"The multiprocessing start method "
                f"{self.MULTI_PROCESSING_START_METHOD.value} "
                f"cannot be used on the Windows platform. "
                f"Only {self.MultiProcessingStartMethod.SPAWN.value} is available."
            )
            raise ValueError(msg)
