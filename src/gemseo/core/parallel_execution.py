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
import os
import queue
import sys
import threading as th
import time
import traceback
from collections.abc import Iterable
from typing import Any
from typing import Callable
from typing import Mapping
from typing import Sequence
from typing import Union

from numpy import ndarray

from gemseo.utils.multiprocessing import get_multi_processing_manager

IS_WIN = os.name == "nt"
ParallelExecutionWorkerType = Union[Sequence[Union[object, Callable]], object, Callable]

SUBPROCESS_NAME = "subprocess"

LOGGER = logging.getLogger(__name__)


def worker(
    par_exe: ParallelExecution | DiscParallelExecution | DiscParallelLinearization,
    queue_in: queue.Queue,
    queue_out: queue.Queue,
) -> None:
    """Execute a function while there are args left in the queue_in.

    Args:
        par_exe: The parallel execution object that contains the function
            to be executed.
        queue_in: The inputs to be evaluated.
        queue_out: The queue object where the outputs of the function will
            be saved.
    """
    for args in iter(queue_in.get, None):
        try:
            sys.stdout.flush()
            task_indx, function_output = par_exe._run_task_by_index(args)
        except Exception as err:
            traceback.print_exc()
            queue_out.put((args, err))
            queue_in.task_done()
            continue
        queue_out.put((task_indx, function_output))
        queue_in.task_done()


class ParallelExecution:
    """Perform a parallel execution of tasks on input values.

    Input values must be a list of independent pointers.
    """

    N_CPUS = mp.cpu_count()

    workers: ParallelExecutionWorkerType
    """The objects that perform the tasks."""

    n_processes: int
    """The maximum simultaneous number of threads or processes."""

    use_threading: bool
    """Whether to use threads instead of processes to parallelize the execution."""

    wait_time_between_fork: float
    """The time to wait between two forks of the process/thread."""

    input_values: ndarray | None
    """The input values to be passed to the workers."""

    def __init__(
        self,
        workers: ParallelExecutionWorkerType,
        n_processes: int = N_CPUS,
        use_threading: bool = False,
        wait_time_between_fork: float = 0.0,
        exceptions_to_re_raise: tuple[type[Exception]] | None = None,
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
            use_threading: If True, use threads instead of processes
                to parallelize the execution.
                Multiprocessing will copy (serialize) all the disciplines,
                while threading will share all the memory.
                This is important to note if you want to execute the same
                discipline multiple times, in which case you shall use
                multiprocessing.
            wait_time_between_fork: The time to wait between two forks of the
                process/thread.
            exceptions_to_re_raise: The exceptions that should be raised again
                when caught inside a worker. If None, all exceptions coming from
                workers are caught and the execution is allowed to continue.

        Raises:
            ValueError: If there are duplicated workers in `workers` when
                using multithreading.
        """  # noqa: D205, D212, D415
        self.workers = workers
        self.n_processes = n_processes
        self.use_threading = use_threading
        if exceptions_to_re_raise is None:
            self.__exceptions_to_re_raise = ()
        else:
            self.__exceptions_to_re_raise = exceptions_to_re_raise

        if use_threading:
            ids = {id(worker) for worker in workers}
            if len(ids) != len(workers):
                raise ValueError(
                    "When using multithreading, all workers"
                    " shall be different objects !"
                )
        self.wait_time_between_fork = wait_time_between_fork
        self.input_values = None

    def _run_task_by_index(self, task_index: int) -> tuple[int, Any]:
        """Run a task from an index of discipline and the input local data.

        The purpose is to be used by multiprocessing queues as a task.

        Args:
            task_index: The index of the task among `self.workers`.

        Returns:
            The task index and the output of its computation.
        """
        input_loc = self.input_values[task_index]
        if ParallelExecution._is_worker(self.workers):
            worker = self.workers
        elif len(self.workers) > 1:
            worker = self.workers[task_index]
        else:
            worker = self.workers[0]

        # return the worker index to order the outputs properly
        output = self._run_task(worker, input_loc)
        return task_index, output

    def execute(
        self,
        input_values: Sequence[ndarray] | ndarray,
        exec_callback: Callable[[int, Any], Any] | None = None,
        task_submitted_callback: Callable | None = None,
    ) -> dict[int, Any]:
        """Execute all the processes.

        Args:
            input_values: The input values.
            exec_callback: A callback function called with the
                pair (index, outputs) as arguments when an item is retrieved
                from the processing. Index is the associated index
                in input_values of the input used to compute the outputs.
                If None, no function is called.
            task_submitted_callback: A callback function called when all the
                tasks are submitted, but not done yet. If None, no function
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
        n_tasks = len(input_values)
        self.input_values = input_values

        if exec_callback is not None and not callable(exec_callback):
            raise TypeError("exec_callback function must be callable !")

        if task_submitted_callback is not None:
            if not callable(task_submitted_callback):
                raise TypeError("task_submitted_callback function must be callable !")

        tasks = list(range(n_tasks))[::-1]
        # Queue for workers
        if self.use_threading:
            queue_in = queue.Queue()
            queue_out = queue.Queue()
        else:
            manager = get_multi_processing_manager()
            queue_in = manager.Queue()
            queue_out = manager.Queue()
            tasks = manager.list(tasks)
        processes = []

        if self.use_threading:
            for _ in range(self.n_processes):
                thread = th.Thread(
                    target=worker,
                    args=(self, queue_in, queue_out),
                    name=SUBPROCESS_NAME,
                )
                thread.daemon = True
                thread.start()
                processes.append(thread)

        else:
            for _ in range(self.n_processes):
                proc = mp.Process(
                    target=worker,
                    args=(self, queue_in, queue_out),
                    name=SUBPROCESS_NAME,
                )
                proc.daemon = True
                proc.start()
                processes.append(proc)

        if mp.current_process().name != SUBPROCESS_NAME or self.use_threading:
            # fill input queue
            while tasks:
                # if not self.use_threading:
                #    lock.acquire()
                task_indx = tasks[-1]
                del tasks[-1]
                # delay the next processes execution after the first one
                if self.wait_time_between_fork > 0 and task_indx > 0:
                    time.sleep(self.wait_time_between_fork)
                queue_in.put(task_indx)

            if task_submitted_callback is not None:
                task_submitted_callback()
                # print("Submitted all tasks in", time.time() - t1)
            # sort the outputs with the same order as functions
            ordered_outputs = [None] * n_tasks
            got_n_outs = 0
            # Retrieve outputs on the fly to call the callbacks, typically
            # iterates progress bar and stores the data in database or cache
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
                    # Call the callback function
                    if exec_callback is not None:
                        exec_callback(index, output)
                got_n_outs += 1

            # Tell threads and processes to terminate
            for _ in processes:
                queue_in.put(None)

            # Join processes and threads
            for proc in processes:
                proc.join()

            # Check for exceptions and eventually raise them if required.
            if isinstance(output, self.__exceptions_to_re_raise):
                raise output

            # Update self.workers objects.
            self._update_local_objects(ordered_outputs)

            # Filter outputs, eventually.
            return self._filter_ordered_outputs(ordered_outputs)

    @staticmethod
    def _filter_ordered_outputs(ordered_outputs):
        """Filters the ordered_outputs.

        Eventually return a subset in the execute method.
        To be overloaded by subclasses.

        Args:
            ordered_outputs: The outputs, map of ``_run_task`` over ``inputs_list``.

        Returns:
            The filtered outputs.
        """
        return ordered_outputs

    def _update_local_objects(self, ordered_outputs):
        """Update the local objects from parallel results.

        The ordered_outputs contains the stacked outputs of the function
        _run_task() To be overloaded by subclasses.

        Args:
            ordered_outputs: The outputs, map of ``_run_task`` over ``inputs_list``.
        """

    @staticmethod
    def _run_task(
        worker: ParallelExecutionWorkerType,
        input_loc: Any,
    ) -> Any:
        """Effectively perform the computation.

        To be overloaded by subclasses.

        Args:
            worker: The worker pointer.
            input_loc: The input of the worker.

        Returns:
            The computation of the task.

        Raises:
            TypeError: If the provided worker has the wrong type.
        """
        if not ParallelExecution._is_worker(worker):
            raise TypeError(f"Cannot handle worker: {worker}.")

        if hasattr(worker, "execute"):
            return worker.execute(input_loc)

        return worker(input_loc)

    @staticmethod
    def _is_worker(
        worker: ParallelExecutionWorkerType,
    ) -> bool:
        """Test if the worker is acceptable.

        A `worker` has to be callable or have an "execute" method.

        Args:
            worker: The worker to test.

        Returns:
            Whether the worker is acceptable.
        """
        return hasattr(worker, "execute") or callable(worker)


class DiscParallelExecution(ParallelExecution):
    """Execute disciplines in parallel."""

    def _update_local_objects(self, ordered_outputs: Mapping[int, Any]) -> None:
        """Update the local objects from the parallel results.

        The ordered_outputs contains the stacked outputs of the function
        _run_task()

        Args:
            ordered_outputs: The outputs, map of _run_task
                over inputs_list.
        """
        if not isinstance(self.workers, Iterable) or not len(self.workers) == len(
            self.input_values
        ):
            if IS_WIN and not self.use_threading:
                self.workers.n_calls += len(self.input_values)
            return
        for disc, output in zip(self.workers, ordered_outputs):
            # Update discipline local data
            disc.local_data = output


class DiscParallelLinearization(ParallelExecution):
    """Linearize disciplines in parallel."""

    def _update_local_objects(self, ordered_outputs: Mapping[int, Any]) -> None:
        """Update the local objects from the parallel results.

        The ordered_outputs contains the stacked outputs of the function
        _run_task()

        Args:
            ordered_outputs: The outputs, map of _run_task
                over inputs_list.
        """
        if not isinstance(self.workers, Iterable) or not len(self.workers) == len(
            self.input_values
        ):
            if IS_WIN and not self.use_threading:
                # Only increase the number of calls if the Jacobian was computed.
                if ordered_outputs[0][0]:
                    self.workers.n_calls += len(self.input_values)
                    self.workers.n_calls_linearize += len(self.input_values)
            return

        for disc, output in zip(self.workers, ordered_outputs):
            # Update discipline jacobian
            disc.jac = output[1]
            # Update discipline local data in case of execution
            disc.local_data = output[0]

    @staticmethod
    def _run_task(worker, input_loc):
        """Effectively performs the computation.

        To be overloaded by subclasses

        Args:
            worker: The worker pointer.
            input_loc: The input of the worker.

        Returns:
            The local data of the worker and its Jacobian.
        """
        jac = worker.linearize(input_loc)
        return worker.local_data, jac

    @staticmethod
    def _filter_ordered_outputs(ordered_outputs):
        """Filter the ordered_outputs.

        Eventually return a subset in the execute method.
        To be overloaded by subclasses.

        Args:
            ordered_outputs: The outputs, map of ``_run_task`` over ``inputs_list``.

        Returns:
            The Jacobians.
        """
        # Only keep the jacobians as outputs, dismiss local_data
        return [out[1] for out in ordered_outputs]
