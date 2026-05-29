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
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from multiprocessing import get_context
from multiprocessing import get_start_method
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Generic
from typing import TypeVar

from docstring_inheritance import GoogleDocstringInheritanceMeta
from strenum import StrEnum

from gemseo.utils.constants import N_CPUS
from gemseo.utils.platform import PLATFORM_IS_LINUX
from gemseo.utils.platform import PLATFORM_IS_WINDOWS

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence
    from concurrent.futures import Executor
    from concurrent.futures import Future

LOGGER = logging.getLogger(__name__)


CallbackType = Callable[[int, Any], None]
"""The type of a callback function."""

ArgT = TypeVar("ArgT")
ReturnT = TypeVar("ReturnT")

CallableType = Callable[[ArgT], ReturnT]


class _TaskCallables(Generic[ArgT, ReturnT]):
    """Manage the call of one callable among callables."""

    callables: Sequence[CallableType[ArgT, ReturnT]]
    """The callables."""

    preprocessors: Iterable[Callable[[int], None]]
    """The preprocessors."""

    def __init__(
        self,
        callables: Sequence[CallableType[ArgT, ReturnT]],
        preprocessors: Iterable[Callable[[int], None]],
    ) -> None:
        """
        Args:
            callables: The callables.
            preprocessors: The functions called before the execution,
                whose unique argument is the task index.
        """  # noqa: D205, D212, D415
        self.callables = callables
        self.preprocessors = preprocessors

    def __call__(self, task_index: int, input_: ArgT) -> ReturnT:
        """Call a callable.

        Args:
            task_index: The index of the callable to call.
            input_: The input to pass to the callable.

        Returns:
            The output of callable.
        """
        index = task_index if len(self.callables) > 1 else 0
        for preprocessor in self.preprocessors:
            preprocessor(index)
        return self.callables[index](input_)


def _uninitialized_worker_task_callables(task_index: int, input_: Any) -> Any:
    """Sentinel raised when ``_init_worker`` did not run in the worker."""
    del task_index, input_
    msg = (
        "The worker task callables are not initialized; "
        "_init_worker did not run in this worker."
    )
    raise RuntimeError(msg)


_WORKER_TASK_CALLABLES: Callable[[int, Any], Any] = _uninitialized_worker_task_callables
"""The task callables stashed in the worker on pool initialization.

This module global is per worker process: each
[ProcessPoolExecutor][concurrent.futures.ProcessPoolExecutor] worker has its own
copy, set by ``_init_worker`` during pool startup. The parent process never sets
it (the thread-pool path dispatches through ``task_callables`` directly to keep
parent state untouched and avoid clobbering by a nested process pool's workers
inside the same interpreter).

Stashing the callables once per worker lets every
[executor.submit][concurrent.futures.Executor.submit] ship only the task index and
the input through the dispatch queue, instead of re-pickling the workers and
preprocessors for every task.

With the [`fork`][multiprocessing.get_context] start method this also preserves the
parent's shared-memory references (e.g. [Synchronized][multiprocessing.Value]
counters), because the initializer arguments are inherited through ``fork`` without
going through pickle.
"""


def _init_worker(task_callables: _TaskCallables[Any, Any]) -> None:
    """Stash the task callables into a worker module global."""
    global _WORKER_TASK_CALLABLES
    _WORKER_TASK_CALLABLES = task_callables


def _run_task(task_index: int, input_: Any) -> Any:
    """Run a task from the worker module global."""
    return _WORKER_TASK_CALLABLES(task_index, input_)


class CallableParallelExecution(
    Generic[ArgT, ReturnT],
    metaclass=GoogleDocstringInheritanceMeta,
):
    """Perform a parallel execution of callables.

    The inputs must be independent objects.
    """

    class MultiProcessingStartMethod(StrEnum):
        """The multiprocessing start method."""

        FORK = "fork"
        SPAWN = "spawn"
        FORKSERVER = "forkserver"

    MULTI_PROCESSING_START_METHOD: ClassVar[MultiProcessingStartMethod] = (
        MultiProcessingStartMethod.FORK if PLATFORM_IS_LINUX else get_start_method()
    )
    """The start method used for multiprocessing."""

    workers: Sequence[CallableType[ArgT, ReturnT]]
    """The objects that perform the tasks."""

    n_processes: int
    """The maximum simultaneous number of threads or processes."""

    use_threading: bool
    """Whether to use threads instead of processes to parallelize the execution."""

    wait_time_between_fork: float
    """The time to wait between two forks of the process/thread."""

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
                if `use_threading` is True, or processes otherwise,
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
                when caught inside a worker. If `None`, all exceptions coming from
                workers are caught and the execution is allowed to continue.

        Raises:
            ValueError: If there are duplicated workers in `workers` when
                using multithreading.
        """  # noqa: D205, D212, D415
        self.workers = workers
        self.n_processes = n_processes
        self.use_threading = use_threading
        self.wait_time_between_fork = wait_time_between_fork
        self.__exceptions_to_re_raise = tuple(exceptions_to_re_raise)
        self._check_unicity(workers)
        self.__check_multiprocessing_start_method()

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

    def execute(
        self,
        inputs: Sequence[ArgT],
        exec_callbacks: Iterable[CallbackType] = (),
        task_submitted_callback: Callable[[], None] | None = None,
        preprocessors: Iterable[Callable[[int], None]] = (),
    ) -> list[ReturnT | None]:
        """Execute all the processes.

        Args:
            inputs: The input values.
            exec_callbacks: Callback functions called with the
                pair (index, outputs) as arguments when an item is retrieved
                from the processing. Index is the associated index
                in inputs of the input used to compute the outputs.
                If empty, no function is called.
            task_submitted_callback: A callback function called when all the
                tasks are submitted, but not done yet. If `None`, no function
                is called.
            preprocessors: The functions called before the execution,
                whose unique argument is the task index.

        Returns:
            The computed outputs.

        Warning:
            This class relies on multiprocessing features, it is therefore
            necessary to protect its execution with an `if __name__ == '__main__':`
            statement when working on Windows.
        """
        n_tasks = len(inputs)
        if n_tasks == 0:
            return []

        ordered_outputs: list[ReturnT | None] = [None] * n_tasks
        task_callables = _TaskCallables(self.workers, preprocessors)
        re_raise: Exception | None = None

        with self._build_executor(task_callables, n_tasks) as executor:
            futures: dict[Future[ReturnT], int] = {}
            # For threading, dispatch through ``task_callables`` directly so that
            # concurrently-running parallel pools in the same process do not
            # clobber each other through the worker module global.
            fn = task_callables if self.use_threading else _run_task
            for task_index, input_ in enumerate(inputs):
                # Delay the next processes execution after the first one.
                if self.wait_time_between_fork > 0 and task_index > 0:
                    time.sleep(self.wait_time_between_fork)
                future = executor.submit(fn, task_index, input_)
                futures[future] = task_index

            if task_submitted_callback is not None:
                task_submitted_callback()

            for future in as_completed(futures):
                index = futures[future]
                try:
                    output = future.result()
                except Exception as err:
                    LOGGER.exception("Failed to execute task indexed %s", index)
                    # Stop the execution only for required exceptions.
                    # Otherwise, keep retrieving the remaining outputs.
                    if isinstance(err, self.__exceptions_to_re_raise):
                        re_raise = err
                        for pending in futures:
                            pending.cancel()
                        break
                else:
                    ordered_outputs[index] = output
                    for callback in exec_callbacks:
                        callback(index, output)

        if re_raise is not None:
            raise re_raise

        return ordered_outputs

    def _build_executor(
        self, task_callables: _TaskCallables[ArgT, ReturnT], n_tasks: int
    ) -> Executor:
        """Return the executor used to dispatch the tasks.

        Args:
            task_callables: The task callables stashed in the worker on initialization.
            n_tasks: The number of tasks to dispatch.

        Returns:
            The configured executor.
        """
        # Never spawn more workers than tasks: extra workers cost a process
        # creation and a pickling of ``task_callables`` (via ``initargs``) for no work.
        max_workers = min(n_tasks, self.n_processes)
        if self.use_threading:
            return ThreadPoolExecutor(max_workers=max_workers)
        return ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=get_context(method=self.MULTI_PROCESSING_START_METHOD),
            initializer=_init_worker,
            initargs=(task_callables,),
        )

    def __check_multiprocessing_start_method(self) -> None:
        """Check the multiprocessing start method with respect to the platform.

        Raises:
            ValueError: If the start method is different from `spawn` on
                Windows platform.
        """
        if (
            not self.use_threading
            and PLATFORM_IS_WINDOWS
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
