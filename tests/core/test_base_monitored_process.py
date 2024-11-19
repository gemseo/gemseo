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

import pytest

from gemseo.core._base_monitored_process import BaseMonitoredProcess
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


def test_execute_monitored(process, monkeypatch):
    """Verify _execute_monitored."""
    monkeypatch.setattr(timer, "perf_counter", SleepingCounter(0.1))
    assert process.execution_statistics.n_executions == 0
    assert process.execution_statistics.n_linearizations == 0
    assert process.execution_statistics.duration == 0.0
    assert process.execution_status.value == ExecutionStatus.Status.DONE
    process._execute_monitored()
    assert process.execution_status.value == ExecutionStatus.Status.DONE
    assert process.execution_statistics.n_executions == 1
    assert process.execution_statistics.n_linearizations == 0
    assert process.execution_statistics.duration == 0.1
