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
from __future__ import annotations

from typing import Final
from unittest.mock import MagicMock
from unittest.mock import PropertyMock
from unittest.mock import patch

import pytest

from gemseo.core._base_monitored_process import BaseMonitoredProcess
from gemseo.core.execution_statistics import ExecutionStatistics
from gemseo.core.execution_status import ExecutionStatus
from gemseo.utils import timer  # noqa: E402
from gemseo.utils.testing.helpers import concretize_classes
from gemseo.utils.testing.mocks import SleepingCounter

NAME: Final[str] = "name"


@pytest.fixture
def process():
    with concretize_classes(BaseMonitoredProcess):
        return BaseMonitoredProcess(NAME)


def test_init(process):
    """Verify __init__."""
    assert process.name == NAME


def test_str(process):
    """Verify __str__."""
    assert str(process) == NAME


@pytest.mark.parametrize("enable_statistics", [True, False])
@pytest.mark.parametrize("enable_status", [True, False])
def test_execute_monitored(process, monkeypatch, enable_statistics, enable_status):
    """Verify _execute_monitored."""
    ExecutionStatistics.is_enabled = enable_statistics
    ExecutionStatus.is_enabled = enable_status

    if enable_statistics:
        monkeypatch.setattr(timer, "perf_counter", SleepingCounter(0.1))
        assert process.execution_statistics.n_executions == 0
        assert process.execution_statistics.n_linearizations == 0
        assert process.execution_statistics.duration == 0.0
    else:
        assert process.execution_statistics.n_executions is None
        assert process.execution_statistics.n_linearizations is None
        assert process.execution_statistics.duration is None
    assert process.execution_status.value == ExecutionStatus.Status.DONE

    if not enable_status:
        # Set a status that should raise an error when status is enabled.
        process.execution_status.value = ExecutionStatus.Status.FAILED

    process._execute_monitored()

    if enable_status:
        assert process.execution_status.value == ExecutionStatus.Status.DONE
    else:
        # The bad status has not changed.
        process.execution_status.value = ExecutionStatus.Status.FAILED

    if enable_statistics:
        assert process.execution_statistics.n_executions == 1
        assert process.execution_statistics.n_linearizations == 0
        assert process.execution_statistics.duration == 0.1
    else:
        assert process.execution_statistics.n_executions is None
        assert process.execution_statistics.n_linearizations is None
        assert process.execution_statistics.duration is None


@pytest.mark.parametrize("enable_statistics", [True, False])
@pytest.mark.parametrize("enable_status", [True, False])
def test_call_monitored(process, enable_statistics, enable_status):
    """Verify _call_monitored."""
    ExecutionStatistics.is_enabled = enable_statistics
    ExecutionStatus.is_enabled = enable_status

    callable_ = MagicMock()
    statistics_recorder = MagicMock()

    with patch(
        "gemseo.core.execution_status.ExecutionStatus.value",
        new_callable=PropertyMock,
    ) as mock_value:
        process._call_monitored(
            callable_, ExecutionStatus.Status.RUNNING, statistics_recorder
        )
        if enable_status:
            mock_value.assert_called()
        else:
            mock_value.assert_not_called()

    if enable_statistics:
        # The callable_ is not called by the statistics_recorder mock.
        callable_.assert_not_called()
        statistics_recorder.assert_called()
    else:
        callable_.assert_called()
        statistics_recorder.assert_not_called()
