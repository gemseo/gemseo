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

from time import sleep
from typing import Final

import pytest

from gemseo.core._base_monitored_process import BaseMonitoredProcess
from gemseo.core.execution_status import ExecutionStatus
from gemseo.utils.compatibility.python import PYTHON_VERSION
from gemseo.utils.platform import PLATFORM_IS_WINDOWS
from gemseo.utils.testing.helpers import concretize_classes

NAME: Final[str] = "name"

if PLATFORM_IS_WINDOWS and PYTHON_VERSION < (  # noqa: SIM108
    3,
    10,
):  # pragma: >=3.10 no cover
    # Workaround sleep that can be shorter than expected, see
    # https://docs.python.org/3.9/library/time.html#time.sleep
    SLEEP_TIME = 0.11
else:
    SLEEP_TIME = 0.1


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


def test_execute_monitored(process):
    """Verify _execute_monitored."""
    assert process.execution_statistics.n_calls == 0
    assert process.execution_statistics.n_calls_linearize == 0
    assert process.execution_statistics.duration == 0.0
    assert process.execution_status.value == ExecutionStatus.Status.PENDING

    process._run = lambda: sleep(SLEEP_TIME)
    process._execute_monitored()

    assert process.execution_status.value == ExecutionStatus.Status.DONE
    assert process.execution_statistics.n_calls == 1
    assert process.execution_statistics.n_calls_linearize == 0
    assert process.execution_statistics.duration > 0.1
