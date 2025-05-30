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
"""The retry discipline."""

from __future__ import annotations

import math
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from ctypes import c_long
from ctypes import py_object
from ctypes import pythonapi
from logging import getLogger
from typing import TYPE_CHECKING

import psutil

from gemseo.core.discipline import Discipline
from gemseo.core.execution_status import ExecutionStatus

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.typing import StrKeyMapping

LOGGER = getLogger(__name__)


class RetryDiscipline(Discipline):
    """A discipline to be executed with retry and timeout options.

    This :class:`.Discipline` wraps another discipline so it can be executed multiple
    times (up to a specified number of trials) if the previous attempts fail to
    produce any result.

    A timeout in seconds can be specified to prevent executions from becoming stuck.
    The timeout can be handled either via a thread or a subprocess. By default, it is
    handled by a thread, this can be changed by setting the ``timeout_with_process``
    argument to ``True``.
    Use a thread when the discipline does not create other processes, otherwise those
    processes will keep running after the timeout duration.
    Beware that using a process is slower, especially under Windows.

    Users can also provide a tuple of :class:`.Exception` that, if one of them is
    raised, it does not retry the execution.

    Please note that the ``TimeoutError`` exception is also caught if the wrapped
    discipline raises such an exception (i.e. aside from ``RetryDiscipline`` itself).
    So it could lead to 2 surprising cases, but in fact normal cases:
    - a ``TimeoutError`` exception even though the user didn't provide any timeout
    value.
    - a ``TimeoutError`` raised sooner than the ``timeout`` value set by the user.
    """

    __n_executions: int
    """The number of performed executions of the discipline."""

    n_trials: int
    """The number of trials to execute the discipline."""

    wait_time: float
    """The time to wait between 2 trials (in seconds)."""

    timeout: float
    """The maximum duration, in seconds, that the discipline is allowed to run."""

    fatal_exceptions: Iterable[type[Exception]]
    """The exceptions for which the code raises an exception and exit immediately
    without retrying a run."""

    def __init__(
        self,
        discipline: Discipline,
        n_trials: int = 5,
        wait_time: float = 0.0,
        timeout: float = math.inf,
        fatal_exceptions: Iterable[type[Exception]] = (),
        timeout_with_process: bool = False,
    ) -> None:
        """
        Args:
            discipline: The discipline to wrap in the retry loop.
            n_trials: The number of trials of the discipline.
            wait_time: The time to wait between 2 trials (in seconds).
            timeout: The maximum duration, in seconds, that the discipline is
                allowed to run. If this time limit is exceeded, the
                execution is terminated. If ``math.inf``, the
                discipline is executed without timeout limit.
            fatal_exceptions: The exceptions for which the code raises an
                exception and exit immediately without retrying a run.
            timeout_with_process: Whether to use a process or a thread when using the
                timeout feature.
        """  # noqa:D205 D212 D415
        super().__init__(discipline.name)
        self._discipline = discipline
        self.__n_executions = 0
        self.io.input_grammar = discipline.io.input_grammar
        self.io.output_grammar = discipline.io.output_grammar
        self.n_trials = n_trials
        self.wait_time = wait_time
        self.timeout = timeout
        self.fatal_exceptions = fatal_exceptions
        self.timeout_with_process = timeout_with_process

    @property
    def n_executions(self) -> int:
        """The number of times the discipline has been retried during execution."""
        return self.__n_executions

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        self.__n_executions = 0

        for n_trial in range(1, self.n_trials + 1):
            self.__n_executions += 1

            LOGGER.debug(
                "Trying to execute the discipline: attempt %d/%d",
                n_trial,
                self.n_trials,
            )

            try:
                if math.isinf(self.timeout):
                    return self._discipline.execute(input_data)
                return self._run_discipline_with_timeout(input_data)

            except FutureTimeoutError:
                msg = (
                    "Timeout reached during the execution of "
                    f"discipline {self._discipline.name}"
                )
                LOGGER.debug(msg)
                current_error = TimeoutError(msg)

            except Exception as error:  # noqa: BLE001
                if isinstance(error, tuple(self.fatal_exceptions)):
                    LOGGER.info(
                        "Failed to execute discipline %s, "
                        "aborting retry because of the exception type %s.",
                        self._discipline.name,
                        type(error),
                    )
                    raise
                current_error = error

            self._discipline.execution_status.value = ExecutionStatus.Status.DONE
            time.sleep(self.wait_time)
            self._run_before_next_trial()

        plural_suffix = "s" if self.n_trials > 1 else ""
        LOGGER.error(
            "Failed to execute discipline %s after %d attempt%s.",
            self._discipline.name,
            self.n_trials,
            plural_suffix,
        )

        raise current_error

    def _run_before_next_trial(self) -> None:
        """Run before the next trial.

        This method is called whenever a trial has just ended, without success.
        It can be used to perform any necessary cleanup or
        preparation for the next trial.
        """

    def _run_discipline_with_timeout(self, input_data: StrKeyMapping) -> StrKeyMapping:
        """Run the discipline with a timeout.

        Args:
            input_data: The input data passed to the discipline.

        Returns:
            The output returned by the discipline.

        Raises:
            FutureTimeoutError: If the execution runs longer than the specified timeout.
        """
        executor_class = (
            ProcessPoolExecutor if self.timeout_with_process else ThreadPoolExecutor
        )

        with executor_class(max_workers=1) as executor:
            run_discipline = executor.submit(
                self._discipline.execute,
                input_data,
            )

            try:
                return run_discipline.result(timeout=self.timeout)
            except FutureTimeoutError:
                if self.timeout_with_process:
                    # Terminate the unique process and its subprocesses.
                    process_id = next(iter(executor._processes.keys()))
                    process = psutil.Process(process_id)
                    for child_process in process.children():
                        child_process.terminate()
                    process.terminate()
                else:
                    # The unique thread is still alive, so we need to kill it.
                    thread = next(iter(executor._threads))
                    # The following is inspired from https://tomerfiliba.com/recipes/Thread2
                    thread_id = c_long(thread.ident)
                    res = pythonapi.PyThreadState_SetAsyncExc(
                        thread_id, py_object(SystemExit)
                    )
                    if res == 0:  # pragma: no cover
                        LOGGER.debug("Invalid thread id %s", thread_id)
                    if res != 1:  # pragma: no cover
                        # If it returns a number greater than one, you're in trouble,
                        # and you should call it again with exc=NULL
                        # to revert the effect.
                        pythonapi.PyThreadState_SetAsyncExc(thread_id, None)
                raise
