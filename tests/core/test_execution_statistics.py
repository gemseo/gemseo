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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                         documentation
#        :author: Damien Guenot
#                 Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

from time import sleep

import pytest

from gemseo.core.execution_statistics import ExecutionStatistics

from .test_base_monitored_process import SLEEP_TIME


@pytest.fixture
def reset() -> None:
    """Reset the state of the class attributes."""
    ExecutionStatistics.is_enabled = True
    ExecutionStatistics.is_time_stamps_enabled = False


@pytest.fixture
def execution_statistics(reset) -> ExecutionStatistics:
    """Return an ExecutionStatistics instance."""
    return ExecutionStatistics("dummy")


def test_default_state(reset):
    """Verify the default state of the recordings."""
    assert ExecutionStatistics.is_enabled
    assert not ExecutionStatistics.is_time_stamps_enabled


def test_n_calls(execution_statistics: ExecutionStatistics):
    """Verify n_calls."""
    assert execution_statistics.n_calls == 0
    execution_statistics.n_calls = 1
    assert execution_statistics.n_calls == 1

    ExecutionStatistics.is_enabled = False

    assert execution_statistics.n_calls is None

    match = "The execution statistics of the object named dummy are disabled."
    with pytest.raises(RuntimeError, match=match):
        execution_statistics.n_calls = 1

    # Re-enable to access the kept results.
    ExecutionStatistics.is_enabled = True

    assert execution_statistics.n_calls == 1


def test_n_calls_linearize(execution_statistics: ExecutionStatistics):
    """Verify n_calls_linearize."""
    assert execution_statistics.n_calls_linearize == 0
    execution_statistics.n_calls_linearize = 1
    assert execution_statistics.n_calls_linearize == 1

    ExecutionStatistics.is_enabled = False

    assert execution_statistics.n_calls_linearize is None

    match = "The execution statistics of the object named dummy are disabled."
    with pytest.raises(RuntimeError, match=match):
        execution_statistics.n_calls_linearize = 1

    # Re-enable to access the kept results.
    ExecutionStatistics.is_enabled = True

    assert execution_statistics.n_calls_linearize == 1


def test_duration(execution_statistics: ExecutionStatistics):
    """Verify duration."""
    assert execution_statistics.duration == 0.0
    execution_statistics.duration = 1.0
    assert execution_statistics.duration == 1.0

    ExecutionStatistics.is_enabled = False

    assert execution_statistics.duration is None

    match = "The execution statistics of the object named dummy are disabled."
    with pytest.raises(RuntimeError, match=match):
        execution_statistics.duration = 1.0

    # Re-enable to access the kept results.
    ExecutionStatistics.is_enabled = True

    assert execution_statistics.duration == 1.0


def test_record(execution_statistics: ExecutionStatistics):
    """Verify record."""
    # Without linearization.
    with execution_statistics.record():
        sleep(SLEEP_TIME)

    assert execution_statistics.n_calls == 1
    assert execution_statistics.n_calls_linearize == 0
    assert 0.1 < execution_statistics.duration < 0.2

    # With linearization.
    with execution_statistics.record(linearize=True):
        sleep(SLEEP_TIME)

    assert execution_statistics.n_calls == 1
    assert execution_statistics.n_calls_linearize == 1
    assert 0.2 < execution_statistics.duration < 0.3

    reference_duration = execution_statistics.duration

    # Disable the statistics: the recorded results shall be kept.
    ExecutionStatistics.is_enabled = False

    with execution_statistics.record():
        sleep(SLEEP_TIME)

    with execution_statistics.record(linearize=True):
        sleep(SLEEP_TIME)

    # Re-enable to access the kept results.
    ExecutionStatistics.is_enabled = True

    assert execution_statistics.n_calls == 1
    assert execution_statistics.n_calls_linearize == 1
    assert execution_statistics.duration == reference_duration


from gemseo.utils import timer  # noqa: E402


def test_time_stamps(execution_statistics: ExecutionStatistics, monkeypatch):
    """Verify the time stamps."""

    class PerfCounter:
        def __init__(self):
            self.__total = 0

        def __call__(self):
            self.__total += SLEEP_TIME
            return self.__total

    monkeypatch.setattr(timer, "perf_counter", PerfCounter())

    assert ExecutionStatistics.time_stamps is None

    ExecutionStatistics.is_time_stamps_enabled = True
    assert not ExecutionStatistics.time_stamps
    assert ExecutionStatistics.time_stamps is not None

    with execution_statistics.record():
        sleep(SLEEP_TIME)

    with execution_statistics.record(linearize=True):
        sleep(SLEEP_TIME)

    assert ExecutionStatistics.time_stamps.keys() == ["dummy"]
    all_values = tuple(next(iter(ExecutionStatistics.time_stamps.values())))

    for values in all_values:
        assert values[1] - values[0] == pytest.approx(SLEEP_TIME)

    # Check the linearization flag.
    assert not all_values[0][2]
    assert all_values[1][2]

    # Check the time stamps' consistency, the start time of second record is higher than
    # end time of the first record.
    assert all_values[1][0] > all_values[0][1]

    # Reset to the default value.
    ExecutionStatistics.is_time_stamps_enabled = False
