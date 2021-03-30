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
"""
Parallel execution of disciplines and functions using multiprocessing
*********************************************************************
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import multiprocessing as mp
import os
import queue
import threading as th
import time
import traceback
from builtins import range, str, zip

from future import standard_library

standard_library.install_aliases()

from gemseo import LOGGER


class ParallelExecution(object):
    """Perform a parallel execution of tasks on input values.

    Input values must be a list of independent pointers.
    """

    N_CPUS = mp.cpu_count()

    def __init__(
        self,
        worker_list,
        n_processes=N_CPUS,
        use_threading=False,
        wait_time_between_fork=0,
    ):
        """
        Constructor.

        :param worker_list: list of objects that perform the tasks
        :param n_processes: maximum number of processors on which to run
        :param use_threading: if True, use Threads instead of processes
            to parallelize the execution
            multiprocessing will copy (serialize) all the disciplines,
            while threading will share all the memory
            This is important to note if you want to execute the same
            discipline multiple times, you shall use multiprocessing
        :param wait_time_between_fork: time waited between two forks of the
            process /Thread
        """
        self.worker_list = worker_list
        self.n_processes = n_processes
        self.use_threading = use_threading

        if not self.use_threading and os.name == "nt":
            raise ValueError(
                "Multiprocessing is not currently supported on"
                " Windows. Please try to use multithreading"
                " instead."
            )

        if use_threading:
            ids = set(id(worker) for worker in worker_list)
            if len(ids) != len(worker_list):
                raise ValueError(
                    "When using multithreading, all workers"
                    " shall be different objects !"
                )
        self.wait_time_between_fork = wait_time_between_fork

    def _run_task_by_index(self, worker_index, input_values):
        """Run a task from an index of discipline.

        Method to run a task from an index of discipline and input local data,
        the purpose is to be used by multiprocessing queues as a task.

        :param worker_index: the index of the worker among self.worker_index
        :param input_values: the input values list
        """
        input_loc = input_values[worker_index]
        worker = self.worker_list[worker_index]
        # return the worker index to order the outputs properly
        output = self._run_task(worker, input_loc)
        return worker_index, output

    def execute(
        self, input_data_list, exec_callback=None, task_submitted_callback=None
    ):
        """Execute all processes.

        :param inputs: the input values (list or values)
        :param exec_callback: callback function called with the
            pair (index, outputs) as arguments when an item is retrieved
            from the processing, where index is the associated index
            in input_data_list, of the input used to compute the outputs
        :param task_submitted_callback: callback function called when all the
            tasks are submitted, but not yet done
        """
        n_workers = len(self.worker_list)

        if len(input_data_list) != n_workers:
            raise ValueError(
                "Parallel execution shall be run "
                + "with a list of "
                + "inputs of same size as the list of disciplines"
            )

        if exec_callback is not None and not callable(exec_callback):
            raise TypeError("exec_callback function must be callable !")

        if task_submitted_callback is not None:
            if not callable(task_submitted_callback):
                raise TypeError("task_submitted_callback function must be callable !")

        # Queue for workers
        if self.use_threading:
            queue_in = queue.Queue()
            queue_out = queue.Queue()
        else:
            mananger = mp.Manager()
            queue_in = mananger.Queue()
            queue_out = mananger.Queue()

        def worker():
            """Worker method executes a function while
            there are args left in the queue_in
            """
            for args in iter(queue_in.get, None):
                try:
                    function_output = self._run_task_by_index(*args)
                except Exception as err:
                    traceback.print_exc()
                    queue_out.put((args[0], err))
                    queue_in.task_done()
                    continue
                queue_out.put(function_output)
                queue_in.task_done()

        processes = []
        if self.use_threading:
            for _ in range(self.n_processes):
                thread = th.Thread(target=worker)
                thread.daemon = True
                thread.start()
                processes.append(thread)
        else:
            for _ in range(self.n_processes):

                proc = mp.Process(target=worker)
                proc.daemon = True
                proc.start()
                processes.append(proc)

        # fill input queue
        for worker_index in range(n_workers):
            # delay the next processes execution after the first one
            if self.wait_time_between_fork > 0 and worker_index > 0:
                time.sleep(self.wait_time_between_fork)
            queue_in.put((worker_index, input_data_list))

        if task_submitted_callback is not None:
            task_submitted_callback()
        # sort the outputs with the same order as functions
        ordered_outputs = [None] * n_workers
        got_n_outs = 0
        # Retrieve outputs on the fly to call the callbacks, typically
        # iterates progress bar and stores the data in database or cache
        while got_n_outs != n_workers:
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
        for proc in processes:
            queue_in.put(None)

        # Join processes and threads
        for proc in processes:
            proc.join()

        # Update self.workers objects
        self._update_local_objects(ordered_outputs)

        # Filters outputs, eventually
        filtered_outputs = self._filter_ordered_outputs(ordered_outputs)

        return filtered_outputs

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
    def _run_task(worker, input_loc):
        """Effectively performs the computation.

        To be overloaded by subclasses.

        :param worker: the worker pointes
        :param input_loc: input of the worker
        """
        if hasattr(worker, "execute"):
            return worker.execute(input_loc)
        if callable(worker):
            return worker(input_loc)
        raise TypeError("cannot handle worker")


class DiscParallelExecution(ParallelExecution):
    """Execute disciplines in parallel."""

    def _update_local_objects(self, ordered_outputs):
        """Update the local objects from parallel results.

        The ordered_outputs contains the stacked outputs of the function
        _run_task()

        :param ordered_outputs: the list of outputs, map of _run_task
            over inputs_list
        """
        for disc, output in zip(self.worker_list, ordered_outputs):
            # Update discipline local data
            disc.local_data = output


class DiscParallelLinearization(ParallelExecution):
    """Linearize disciplines in parallel."""

    def _update_local_objects(self, ordered_outputs):
        """Update the local objects from parallel results.

        The ordered_outputs contains the stacked outputs of the function
        _run_task()

        :param ordered_outputs: the list of outputs, map of _run_task
            over inputs_list
        """
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
