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
"""Tests for retry discipline."""

from __future__ import annotations

import math
import shlex
import time
from subprocess import Popen
from time import sleep
from typing import TYPE_CHECKING

import psutil
import pytest
from numpy import array

from gemseo import create_discipline
from gemseo.core.discipline import Discipline
from gemseo.core.execution_status import ExecutionStatus
from gemseo.disciplines.wrappers.retry_discipline import RetryDiscipline
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT
from gemseo.utils.platform import PLATFORM_IS_WINDOWS
from gemseo.utils.testing.helpers import assert_exception
from gemseo.utils.timer import Timer

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from gemseo.typing import StrKeyMapping


@pytest.fixture
def an_analytic_discipline() -> Discipline:
    """Analytic discipline."""
    return create_discipline("AnalyticDiscipline", {"y": "x"})


@pytest.fixture
def a_crashing_analytic_discipline() -> Discipline:
    """Analytic discipline crashing when x=0."""
    return create_discipline("AnalyticDiscipline", {"y": "1.0/x"})


@pytest.fixture
def a_crashing_discipline_in_run() -> Discipline:
    return CrashingDisciplineInRun(name="Crash_run")


class CrashingDisciplineInRun(Discipline):
    """A discipline raising NotImplementedError in `_run`."""

    def _run(self, input_data: StrKeyMapping):
        msg = "Error: This method is not implemented."
        raise NotImplementedError(msg)


class FictiveDiscipline(Discipline):
    """Discipline to be executed several times.

    - The first 2 times, raise a RuntimeError,
    - and finally succeed.
    """

    def __init__(self) -> None:
        super().__init__()
        self.attempt = 0

    def _run(self, input_data: StrKeyMapping) -> None:
        self.attempt += 1
        if self.attempt < 3:
            msg = "runtime error in FictiveDiscipline"
            raise RuntimeError(msg)


@pytest.mark.parametrize("timeout", [math.inf, 10.0])
def test_retry_discipline(an_analytic_discipline, timeout, caplog) -> None:
    """Test discipline, no timeout set."""
    retry_discipline = RetryDiscipline(an_analytic_discipline, timeout=timeout)
    retry_discipline.execute({"x": array([4.0])})

    assert retry_discipline.n_executions == 1
    assert retry_discipline.local_data == {"x": array([4.0]), "y": array([4.0])}
    assert caplog.text == ""


class SlowThreadDiscipline(Discipline):
    """A discipline that executes a long-running ."""

    def _run(self, input_data: StrKeyMapping = READ_ONLY_EMPTY_DICT) -> None:
        # Use sleep instead of a busy-wait loop: sleep releases the GIL, so
        # PyThreadState_SetAsyncExc can deliver SystemExit almost immediately
        # at the next bytecode boundary.  A tight C-level loop (time.time())
        # provides very few such boundaries and makes injection unpredictable.
        time.sleep(60.0)


class SlowProcessDiscipline(Discipline):
    """A discipline that executes a long-running process."""

    pid_path: Path
    """The path to the file to contain the process id."""

    def _run(self, input_data: StrKeyMapping = READ_ONLY_EMPTY_DICT) -> None:
        cmd_line = 'python -c "import time; time.sleep(60.0)"'

        # The following is inspired from subprocess.run.
        process = Popen(shlex.split(cmd_line))
        # Store the process id to later check if it is still running.
        self.pid_path.write_text(str(process.pid))
        try:
            process.communicate()
        except:
            process.kill()
            raise
        process.poll()

        sleep(60)


def test_failure_zero_division_error(
    a_crashing_analytic_discipline, caplog, snapshot
) -> None:
    """Test failure of the discipline with a bad x entry.

    In order to catch the ZeroDivisionError, set n_trials=1
    """
    disc = RetryDiscipline(a_crashing_analytic_discipline, n_trials=1)
    with assert_exception(ZeroDivisionError, snapshot):
        disc.execute({"x": array([0.0])})

    assert disc.local_data == {"x": array([0.0])}
    assert disc.n_executions == 1

    log_message = "Failed to execute discipline AnalyticDiscipline after 1 attempt."
    assert log_message in caplog.text


@pytest.mark.parametrize(
    "fatal_exceptions",
    [
        (ZeroDivisionError,),
        (ZeroDivisionError, FloatingPointError, OverflowError),
        (OverflowError, FloatingPointError, ZeroDivisionError),
    ],
)
@pytest.mark.parametrize("n_trials", [1, 3])
def test_failure_zero_division_error_n_trials(
    n_trials: int,
    fatal_exceptions: Iterable[type[Exception]],
    a_crashing_analytic_discipline,
    caplog,
    snapshot,
) -> None:
    """Test failure of the discipline with timeout and a bad x entry.

    In order to catch the ZeroDivisionError that arises before timeout (5s), test with
    n_trials=1 and 3 to be sure every case is ok.
    """
    disc = RetryDiscipline(
        a_crashing_analytic_discipline,
        n_trials=n_trials,
        fatal_exceptions=fatal_exceptions,
    )
    with assert_exception(ZeroDivisionError, snapshot):
        disc.execute({"x": array([0.0])})

    assert disc.n_executions == 1
    assert disc.local_data == {"x": array([0.0])}

    log_message = (
        "Failed to execute discipline AnalyticDiscipline,"
        " aborting retry because of the exception type <class 'ZeroDivisionError'>."
    )
    assert log_message in caplog.text


def test_a_not_implemented_error_analytic_discipline(
    a_crashing_discipline_in_run,
    caplog,
    snapshot,
) -> None:
    """Test discipline with a_crashing_discipline_in_run and a tuple of

    a tuple of fatal_exceptions that abort the retry (ZeroDivisionError).
    """
    retry_discipline = RetryDiscipline(
        a_crashing_discipline_in_run,
        n_trials=5,
        timeout=100.0,
        fatal_exceptions=(
            ZeroDivisionError,
            FloatingPointError,
            OverflowError,
            NotImplementedError,
        ),
    )
    with assert_exception(NotImplementedError, snapshot):
        retry_discipline.execute({"x": array([1.0])})

    assert retry_discipline.n_executions == 1
    assert retry_discipline.local_data == {}

    log_message = (
        "Failed to execute discipline Crash_run, aborting retry "
        "because of the exception type <class 'NotImplementedError'>."
    )
    assert log_message in caplog.text


@pytest.mark.parametrize("n_trials", [1, 3])
def test_1_3times_failing(
    a_crashing_analytic_discipline, n_trials, caplog, snapshot
) -> None:
    """Test a discipline crashing each time, n_trials = 1 or 3."""
    disc = RetryDiscipline(
        a_crashing_analytic_discipline,
        n_trials=n_trials,
    )
    with assert_exception(ZeroDivisionError, snapshot):
        disc.execute({"x": array([0.0])})

    assert disc.n_executions == n_trials
    assert disc.io.data == {"x": array([0.0])}

    plural_suffix = "s" if n_trials > 1 else ""
    log_message = (
        "Failed to execute discipline AnalyticDiscipline"
        f" after {n_trials} attempt{plural_suffix}."
    )
    assert log_message in caplog.text


def test_2fails_then_succeed() -> None:
    """Test a discipline crashing the 2 first times, then succeeding."""
    disc = RetryDiscipline(
        FictiveDiscipline(),
        n_trials=3,
    )
    disc.execute()
    assert disc.n_executions == 3
    assert disc.io.data == {}
    assert disc.execution_status.value == ExecutionStatus.Status.DONE


THREAD_TIMEOUT = 10.0 if PLATFORM_IS_WINDOWS else 1.0
PROCESS_TIMEOUT = 10.0 if PLATFORM_IS_WINDOWS else 4.0


# TODO: Fix this test on windows.
@pytest.mark.skipif(
    PLATFORM_IS_WINDOWS,
    reason="The windows CI has big troubles with this test.",
)
@pytest.mark.parametrize("wait_time", [0.01, 0.1])
@pytest.mark.parametrize("n_trials", [1, 3])
@pytest.mark.parametrize("disc_class", [SlowProcessDiscipline, SlowThreadDiscipline])
def test_wait_time_and_n_trials(
    n_trials,
    wait_time,
    disc_class,
    caplog,
    tmp_wd,
    snapshot,
) -> None:
    """Test failure of the discipline with a too much very short timeout."""
    wrapped_disc = disc_class()
    wrapped_disc.pid_path = pid_path = tmp_wd / "pid"

    timeout = PROCESS_TIMEOUT if disc_class == SlowProcessDiscipline else THREAD_TIMEOUT
    disc = RetryDiscipline(
        wrapped_disc,
        timeout=timeout,
        n_trials=n_trials,
        wait_time=wait_time,
        timeout_with_process=disc_class == SlowProcessDiscipline,
    )

    with (
        Timer() as timer,
        assert_exception(TimeoutError, snapshot),
    ):
        disc.execute()

    if disc_class == SlowProcessDiscipline:
        # pid_path holds the PID of the subprocess spawned by Popen *inside* the
        # worker process, i.e. a grandchild of the test process.  Verifying it is
        # gone confirms that recursive=True in _terminate_process captured it.
        assert not psutil.pid_exists(int(pid_path.read_text()))

    # The offset accounts for the time it takes for the execution to reach the _run of
    # retry discipline.
    assert timer.elapsed_time > 0.01 + (n_trials - 1) * wait_time
    assert disc.n_executions == n_trials
    assert not disc.local_data

    plural_suffix = "s" if n_trials > 1 else ""
    log_message = (
        f"Failed to execute discipline {disc_class.__name__} after {n_trials}"
        f" attempt{plural_suffix}."
    )
    assert log_message in caplog.text


def test_n_trials_zero_raises(an_analytic_discipline, snapshot) -> None:
    """Test that n_trials=0 raises ValueError immediately."""
    with assert_exception(ValueError, snapshot):
        RetryDiscipline(an_analytic_discipline, n_trials=0)


def test_per_attempt_warning_logged(caplog) -> None:
    """Test that each non-fatal failure emits a WARNING log entry."""
    disc = RetryDiscipline(FictiveDiscipline(), n_trials=3)
    disc.execute()

    # FictiveDiscipline raises RuntimeError on attempts 1 and 2.
    warning_msgs = [
        r.message
        for r in caplog.records
        if r.levelname == "WARNING" and "failed with" in r.message
    ]
    assert len(warning_msgs) == 2


# TODO: Fix this test on windows.
@pytest.mark.skipif(
    PLATFORM_IS_WINDOWS,
    reason="The windows CI has big troubles with this test.",
)
def test_timeout_warning_logged(caplog, snapshot) -> None:
    """Test that a timeout is logged at WARNING level."""
    disc = RetryDiscipline(SlowThreadDiscipline(), timeout=0.1, n_trials=1)
    with assert_exception(TimeoutError, snapshot):
        disc.execute()

    assert any(
        "Timeout reached" in r.message and r.levelname == "WARNING"
        for r in caplog.records
    )


def test_sleep_not_called_when_wait_time_zero(monkeypatch) -> None:
    """Test that time.sleep is not called when wait_time=0.0."""
    sleep_calls = []
    monkeypatch.setattr(
        "gemseo.disciplines.wrappers.retry_discipline.time.sleep",
        sleep_calls.append,
    )
    disc = RetryDiscipline(FictiveDiscipline(), n_trials=3, wait_time=0.0)
    disc.execute()

    assert sleep_calls == []
