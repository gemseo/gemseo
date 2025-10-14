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
import re
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
from gemseo.utils.timer import Timer

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from gemseo.typing import StrKeyMapping


@pytest.fixture
def an_analytic_discipline() -> Discipline:
    """Analytic discipline."""
    return create_discipline("AnalyticDiscipline", expressions={"y": "x"})


@pytest.fixture
def a_crashing_analytic_discipline() -> Discipline:
    """Analytic discipline crashing when x=0."""
    return create_discipline("AnalyticDiscipline", expressions={"y": "1.0/x"})


@pytest.fixture
def a_crashing_discipline_in_run() -> Discipline:
    return CrashingDisciplineInRun(name="Crash_run")


class CrashingDisciplineInRun(Discipline):
    """A discipline raising NotImplementedError in ``_run``."""

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
        end_time = time.time() + 60.0

        while time.time() < end_time:
            pass


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


def test_failure_zero_division_error(a_crashing_analytic_discipline, caplog) -> None:
    """Test failure of the discipline with a bad x entry.

    In order to catch the ZeroDivisionError, set n_trials=1
    """
    disc = RetryDiscipline(a_crashing_analytic_discipline, n_trials=1)
    with pytest.raises(ZeroDivisionError, match="float division by zero"):
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
    with pytest.raises(ZeroDivisionError, match="float division by zero"):
        disc.execute({"x": array([0.0])})

    assert disc.n_executions == 1
    assert disc.local_data == {"x": array([0.0])}

    log_message = (
        "Failed to execute discipline AnalyticDiscipline,"
        " aborting retry because of the exception type <class 'ZeroDivisionError'>."
    )
    assert log_message in caplog.text


def test_a_not_implemented_error_analytic_discipline(
    a_crashing_discipline_in_run, caplog
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
    with pytest.raises(
        NotImplementedError, match=re.escape("Error: This method is not implemented.")
    ):
        retry_discipline.execute({"x": array([1.0])})

    assert retry_discipline.n_executions == 1
    assert retry_discipline.local_data == {}

    log_message = (
        "Failed to execute discipline Crash_run, aborting retry "
        "because of the exception type <class 'NotImplementedError'>."
    )
    assert log_message in caplog.text


@pytest.mark.parametrize("n_trials", [1, 3])
def test_1_3times_failing(a_crashing_analytic_discipline, n_trials, caplog) -> None:
    """Test a discipline crashing each time, n_trials = 1 or 3."""
    disc = RetryDiscipline(
        a_crashing_analytic_discipline,
        n_trials=n_trials,
    )
    with pytest.raises(ZeroDivisionError, match="float division by zero"):
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


TIMEOUT = 10.0 if PLATFORM_IS_WINDOWS else 0.1


# TODO: Fix this test on windows.
@pytest.mark.skipif(
    PLATFORM_IS_WINDOWS,
    reason="The windows CI has big troubles with this test.",
)
@pytest.mark.parametrize("wait_time", [0.01, 0.1])
@pytest.mark.parametrize("n_trials", [1, 3])
@pytest.mark.parametrize("disc_class", [SlowProcessDiscipline, SlowThreadDiscipline])
def test_wait_time_and_n_trials(
    n_trials, wait_time, disc_class, caplog, tmp_wd
) -> None:
    """Test failure of the discipline with a too much very short timeout."""
    wrapped_disc = disc_class()
    wrapped_disc.pid_path = pid_path = tmp_wd / "pid"

    disc = RetryDiscipline(
        wrapped_disc,
        timeout=TIMEOUT,
        n_trials=n_trials,
        wait_time=wait_time,
        timeout_with_process=disc_class == SlowProcessDiscipline,
    )

    with (
        Timer() as timer,
        pytest.raises(
            TimeoutError,
            match="Timeout reached during the execution"
            rf" of discipline {disc_class.__name__}",
        ),
    ):
        disc.execute()

    if disc_class == SlowProcessDiscipline:
        # The wrapped discipline process should have been killed.
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
