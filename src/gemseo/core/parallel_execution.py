# -*- coding: utf-8 -*-
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
#        :author: Charlie Vanaret, Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Parallel execution of disciplines and functions using multiprocessing."""
from __future__ import division, unicode_literals

import logging
import multiprocessing as mp
import os
import queue
import sys
import threading as th
import time
import traceback
from collections import Iterable
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

from numpy import ndarray

IS_WIN = os.name == "nt"
ParallelExecutionWorkerType = Union[Sequence[Union[object, Callable]], object, Callable]

SUBPROCESS_NAME = "subprocess"

LOGGER = logging.getLogger(__name__)


def worker(
    par_exe,  # type: Union[ParallelExecution, DiscParallelExecution, DiscParallelLinearization]
    queue_in,  # type: queue.Queue
    queue_out,  # type: queue.Queue
):  # type: (...) -> None
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


class ParallelExecution(object):
    """Perform a parallel execution of tasks on input values.

    Input values must be a list of independent pointers.
    """

    N_CPUS = mp.cpu_count()

    def __init__(
        self,
        worker_list,  # type: ParallelExecutionWorkerType
        n_processes=N_CPUS,  # type: int
        use_threading=False,  # type: bool
        wait_time_between_fork=0.0,  # type: float
    ):  # type: (...) -> None
        """
        Args:
            worker_list: The objects that perform the tasks.
                Either pass one worker, and it will be forked in multiprocessing.
                Or, when using multithreading or different workers, pass one worker
                per input data.
            n_processes: The maximum number of processes on which to run.
            use_threading: If True, use Threads instead of processes
                to parallelize the execution.
                Multiprocessing will copy (serialize) all the disciplines,
                while threading will share all the memory.
                This is important to note if you want to execute the same
                discipline multiple times, in which case you shall use
                multiprocessing.
            wait_time_between_fork: The time to wait between two forks of the
                process/Thread.

        Raises:
            ValueError: If there are duplicated workers in `worker_list` when
                using multithreading.
        """
        self.worker_list = worker_list
        self.n_processes = n_processes
        self.use_threading = use_threading

        if use_threading:
            ids = set(id(worker) for worker in worker_list)
            if len(ids) != len(worker_list):
                raise ValueError(
                    "When using multithreading, all workers"
                    " shall be different objects !"
                )
        self.wait_time_between_fork = wait_time_between_fork
        self.input_data_list = None

    def _run_task_by_index(
        self, task_index  # type: int
    ):  # type: (...) -> Tuple[int, Any]
        """Run a task from an index of discipline and the input local data.

        The purpose is to be used by multiprocessing queues as a task.

        Args:
            task_index: The index of the task among `self.worker_list`.

        Returns:
            The task index and the output of its computation.
        """
        input_loc = self.input_data_list[task_index]
        if ParallelExecution._is_worker(self.worker_list):
            worker = self.worker_list
        elif len(self.worker_list) > 1:
            worker = self.worker_list[task_index]
        else:
            worker = self.worker_list[0]

        # return the worker index to order the outputs properly
        output = self._run_task(worker, input_loc)
        return task_index, output

    def execute(
        self,
        input_data_list,  # type: Union[Sequence[ndarray], ndarray]
        exec_callback=None,  # type: Optional[Callable[[int, Any], Any]]
        task_submitted_callback=None,  # type: Optional[Callable]
    ):  # type: (...) -> Dict[int, Any]
        """Execute all the processes.

        Args:
            input_data_list: The input values.
            exec_callback: A callback function called with the
                pair (index, outputs) as arguments when an item is retrieved
                from the processing. Index is the associated index
                in input_data_list of the input used to compute the outputs.
                If None, no function is called.
            task_submitted_callback: A callback function called when all the
                tasks are submitted, but not done yet. If None, no function
                is called.

        Returns:
            The computed outputs.

        Raises:
            TypeError: If the `exec_callback` is not callable.
                If the `task_submitted_callback` is not callable.
        """

        n_tasks = len(input_data_list)
        self.input_data_list = input_data_list

        if exec_callback is not None and not callable(exec_callback):
            raise TypeError("exec_callback function must be callable !")

        if task_submitted_callback is not None:
            if not callable(task_submitted_callback):
                raise TypeError("task_submitted_callback function must be callable !")

        tasks_list = list(range(n_tasks))[::-1]
        # Queue for workers
        if self.use_threading:
            queue_in = queue.Queue()
            queue_out = queue.Queue()
        else:
            mananger = mp.Manager()
            queue_in = mananger.Queue()
            queue_out = mananger.Queue()
            tasks_list = mananger.list(tasks_list)
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
            while tasks_list:
                # if not self.use_threading:
                #    lock.acquire()
                task_indx = tasks_list[-1]
                del tasks_list[-1]
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
            while got_n_outs != n_tasks:
                index, output = queue_out.get()
                if isinstance(output, Exception):
                    LOGGER.error("Failed to execute task indexed %s", str(index))
                    LOGGER.error(output)
                else:
                    ordered_outputs[index] = output
                    # Call the callback function
                    if exec_callback is not None:
                        exec_callback(index, output)
                got_n_outs += 1

            # Tells threads and processes to terminate
            for _ in processes:
                queue_in.put(None)

            # Join processes and threads
            for proc in processes:
                proc.join()

            # Update self.workers objects
            self._update_local_objects(ordered_outputs)

            # Filters outputs, eventually
            return self._filter_ordered_outputs(ordered_outputs)

    @staticmethod
    def _filter_ordered_outputs(ordered_outputs):
        """Filters the ordered_outputs.

        Eventually return a subset in the execute method.
        To be overloaded by subclasses.

        :param ordered_outputs: the list of outputs, map of _run_task
           over inputs_list
        """
        return ordered_outputs

    def _update_local_objects(self, ordered_outputs):
        """Update the local objects from parallel results.

        The ordered_outputs contains the stacked outputs of the function
        _run_task() To be overloaded by subclasses.

        :param ordered_outputs: the list of outputs, map of _run_task
            over inputs_list
        """

    @staticmethod
    def _run_task(
        worker,  # type: ParallelExecutionWorkerType
        input_loc,  # type: Any
    ):  # type: (...) -> Any
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
            raise TypeError("Cannot handle worker: {}.".format(worker))

        if hasattr(worker, "execute"):
            return worker.execute(input_loc)

        return worker(input_loc)

    @staticmethod
    def _is_worker(
        worker,  # type: ParallelExecutionWorkerType
    ):  # type: (...) -> bool
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

    def _update_local_objects(
        self, ordered_outputs  # type: Mapping[int, Any]
    ):  # type: (...) -> None
        """Update the local objects from the parallel results.

        The ordered_outputs contains the stacked outputs of the function
        _run_task()

        Args:
            ordered_outputs: The outputs, map of _run_task
                over inputs_list.
        """
        if not isinstance(self.worker_list, Iterable) or not len(
            self.worker_list
        ) == len(self.input_data_list):
            if IS_WIN and not self.use_threading:
                self.worker_list.n_calls += len(self.input_data_list)
            return
        for disc, output in zip(self.worker_list, ordered_outputs):
            # Update discipline local data
            disc.local_data = output


class DiscParallelLinearization(ParallelExecution):
    """Linearize disciplines in parallel."""

    def _update_local_objects(
        self, ordered_outputs  # type: Mapping[int, Any]
    ):  # type: (...) -> None
        """Update the local objects from the parallel results.

        The ordered_outputs contains the stacked outputs of the function
        _run_task()

        Args:
            ordered_outputs: The outputs, map of _run_task
                over inputs_list.
        """
        if not isinstance(self.worker_list, Iterable) or not len(
            self.worker_list
        ) == len(self.input_data_list):
            if IS_WIN and not self.use_threading:
                # Only increase the number of calls if the Jacobian was computed.
                if ordered_outputs[0][0]:
                    self.worker_list.n_calls += len(self.input_data_list)
                    self.worker_list.n_calls_linearize += len(self.input_data_list)
            return

        for disc, output in zip(self.worker_list, ordered_outputs):
            # Update discipline jacobian
            disc.jac = output[1]
            # Update discipline local data in case of execution
            disc.local_data = output[0]

    @staticmethod
    def _run_task(worker, input_loc):
        """Effectively performs the computation.

        To be overloaded by subclasses

        :param worker: the worker pointes
        :param input_loc: input of the worker
        """
        jac = worker.linearize(input_loc)
        return worker.local_data, jac

    @staticmethod
    def _filter_ordered_outputs(ordered_outputs):
        """Filters the ordered_outputs.

        Eventually return a subset in the execute method.
        To be overloaded by subclasses.

        :param ordered_outputs: the list of outputs, map of _run_task
           over inputs_list
        """
        # Only keep the jacobians as outputs, dismiss local_data
        return [out[1] for out in ordered_outputs]
