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
import time
from typing import TYPE_CHECKING

import pytest
from numpy import array

from gemseo import create_discipline
from gemseo.core.discipline import Discipline
from gemseo.core.execution_status import ExecutionStatus
from gemseo.disciplines.wrappers.retry_discipline import RetryDiscipline
from gemseo.utils.timer import Timer

if TYPE_CHECKING:
    from collections.abc import Iterable

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


@pytest.fixture
def a_long_time_running_discipline() -> Discipline:
    return DisciplineLongTimeRunning()


class CrashingDisciplineInRun(Discipline):
    """A discipline raising NotImplementedError in ``_run``."""

    def _run(self, input_data: StrKeyMapping):
        msg = "Error: This method is not implemented."
        raise NotImplementedError(msg)


class DisciplineLongTimeRunning(Discipline):
    """A discipline that could run for a while, to test the timeout feature."""

    def _run(self, input_data: StrKeyMapping) -> None:
        time.sleep(5.0)


class FictiveDiscipline(Discipline):
    """Discipline to be executed several times.

    - The first 2 times, raise a RuntimeError,
    - and finally succeed.
    """

    def __init__(self) -> None:
        super().__init__()
        self.attempt = 0

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping:
        self.attempt += 1
        if self.attempt < 3:
            msg = "runtime error in FictiveDiscipline"
            raise RuntimeError(msg)
        return {}


@pytest.mark.parametrize("timeout", [math.inf, 10.0])
def test_retry_discipline(an_analytic_discipline, timeout, caplog) -> None:
    """Test discipline, no timeout set."""
    retry_discipline = RetryDiscipline(an_analytic_discipline, timeout=timeout)
    retry_discipline.execute({"x": array([4.0])})

    assert retry_discipline.n_executions == 1
    assert retry_discipline.local_data == {"x": array([4.0]), "y": array([4.0])}

    assert caplog.text == ""


@pytest.mark.parametrize("wait_time", [0.5, 1.0])
@pytest.mark.parametrize("n_trials", [1, 3])
def test_failure_retry_discipline_with_timeout(
    an_analytic_discipline, n_trials, wait_time, caplog
) -> None:
    """Test failure of the discipline with a too much very short timeout."""
    disc_with_timeout = RetryDiscipline(
        an_analytic_discipline, timeout=1e-4, n_trials=n_trials, wait_time=wait_time
    )

    with (
        Timer() as timer,
        pytest.raises(
            TimeoutError,
            match="Timeout reached during the execution"
            " of discipline AnalyticDiscipline",
        ),
    ):
        disc_with_timeout.execute({"x": array([4.0])})

    elapsed_time = timer.elapsed_time
    assert elapsed_time > 0.05 + (n_trials - 1) * wait_time

    assert disc_with_timeout.n_executions == n_trials
    assert disc_with_timeout.local_data == {"x": array([4.0])}

    assert "Process stopped as it exceeds timeout" in caplog.text

    plural_suffix = "s" if n_trials > 1 else ""
    log_message = (
        f"Failed to execute discipline AnalyticDiscipline after {n_trials}"
        f" attempt{plural_suffix}."
    )
    assert log_message in caplog.text


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
def test_failure_zero_division_error_with_timeout(
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
        timeout=10.0,
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


def test_retry_discipline_timeout_feature(
    a_long_time_running_discipline, caplog
) -> None:
    """Test the timeout feature of discipline with a long computation."""
    n_trials = 1

    disc_with_timeout = RetryDiscipline(
        a_long_time_running_discipline, timeout=2.0, n_trials=n_trials
    )
    with pytest.raises(
        TimeoutError,
        match="Timeout reached during the execution"
        " of discipline DisciplineLongTimeRunning",
    ):
        disc_with_timeout.execute({"x": array([0.0])})

    assert disc_with_timeout.n_executions == n_trials
    assert disc_with_timeout.local_data == {}

    assert "Process stopped as it exceeds timeout" in caplog.text
    log_message = (
        "Failed to execute discipline DisciplineLongTimeRunning after 1 attempt."
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

    assert disc.execution_status.value == ExecutionStatus.Status.FAILED

    plural_suffix = "s" if n_trials > 1 else ""
    log_message = (
        "Failed to execute discipline AnalyticDiscipline"
        f" after {n_trials} attempt{plural_suffix}."
    )
    assert log_message in caplog.text


def test_2fails_then_succeed(caplog) -> None:
    """Test a discipline crashing the 2 first times, then succeeding."""
    disc = RetryDiscipline(
        FictiveDiscipline(),
        n_trials=3,
    )
    disc.execute()
    assert disc.n_executions == 3
    assert disc.io.data == {}
    assert disc.execution_status.value == ExecutionStatus.Status.DONE
