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

from typing import TYPE_CHECKING

import pytest

from gemseo.core.base_execution_status_observer import BaseExecutionStatusObserver
from gemseo.core.execution_status import ExecutionStatus
from gemseo.utils.pickle import from_pickle
from gemseo.utils.pickle import to_pickle

if TYPE_CHECKING:
    from collections.abc import Iterable


Status = ExecutionStatus.Status


@pytest.fixture
def execution_status() -> ExecutionStatus:
    return ExecutionStatus("")


def test_initial_status(execution_status: ExecutionStatus):
    """Verify the initial status."""
    assert execution_status.value == Status.DONE


INITIAL_TO_NEW_STATUSES_WITH_ERRORS = {
    Status.LINEARIZING: {Status.RUNNING, Status.LINEARIZING},
    Status.FAILED: {Status.RUNNING, Status.LINEARIZING},
    Status.RUNNING: {Status.RUNNING, Status.LINEARIZING},
}


@pytest.mark.parametrize(
    ("initial_status", "new_statuses"),
    [
        # TODO: should be only  RUNNING or LINEARIZING
        (Status.DONE, Status),
        (
            Status.LINEARIZING,
            set(Status) - INITIAL_TO_NEW_STATUSES_WITH_ERRORS[Status.LINEARIZING],
        ),
        (
            Status.FAILED,
            set(Status) - INITIAL_TO_NEW_STATUSES_WITH_ERRORS[Status.FAILED],
        ),
        (
            Status.RUNNING,
            set(Status) - INITIAL_TO_NEW_STATUSES_WITH_ERRORS[Status.RUNNING],
        ),
    ],
)
def test_value_getter_setter(initial_status: Status, new_statuses: Iterable[Status]):
    """Verify the getter and setter without errors."""
    for new_status in new_statuses:
        execution_status = ExecutionStatus("")
        execution_status.value = initial_status
        assert execution_status.value == initial_status
        execution_status.value = new_status
        assert execution_status.value == new_status


@pytest.mark.parametrize(
    ("initial_status", "new_statuses"), INITIAL_TO_NEW_STATUSES_WITH_ERRORS.items()
)
def test_value_setter_errors(initial_status: Status, new_statuses: Iterable[Status]):
    """Verify the setter with errors."""
    for new_status in new_statuses:
        execution_status = ExecutionStatus("")
        execution_status.value = initial_status
        match = (
            f" cannot be set to status {new_status} while in status {initial_status}"
        )
        with pytest.raises(ValueError, match=match):
            execution_status.value = new_status


def test_value_setter_error_without_enum(execution_status: ExecutionStatus):
    """Verify the setter error alien status."""
    match = "'bad' is not a valid ExecutionStatus.Status"
    with pytest.raises(ValueError, match=match):
        execution_status.value = "bad"


@pytest.mark.parametrize("status", [Status.RUNNING, Status.LINEARIZING])
def test_handle_success(execution_status: ExecutionStatus, status: Status):
    """Verify the handle method on success."""
    execution_status.value = Status.DONE
    execution_status.handle(status, lambda: None)
    assert execution_status.value == Status.DONE


@pytest.mark.parametrize("status", [Status.RUNNING, Status.LINEARIZING])
def test_handle_failure(execution_status: ExecutionStatus, status: Status):
    """Verify the handle method on failure."""
    execution_status.value = Status.DONE

    def _raise(msg: str) -> None:
        raise ValueError(msg)

    with pytest.raises(ValueError, match="message"):
        execution_status.handle(status, _raise, "message")
    assert execution_status.value == Status.FAILED


@pytest.mark.parametrize("status", [Status.RUNNING, Status.LINEARIZING])
def test_handle_bad_initial_status(execution_status: ExecutionStatus, status: Status):
    """Verify the handle method on bad initial status."""
    execution_status.value = Status.LINEARIZING
    match = f" cannot be set to status {status} while in status LINEARIZING"
    with pytest.raises(ValueError, match=match):
        execution_status.handle(status, lambda: None)
    assert execution_status.value == Status.LINEARIZING


class Observer(BaseExecutionStatusObserver):
    def __init__(self) -> None:
        self.status = None

    def update_status(self, execution_status: ExecutionStatus) -> None:
        self.status = execution_status.value


def test_observers(execution_status: ExecutionStatus):
    """Verify observers."""
    observer1 = Observer()
    assert observer1.status is None
    execution_status.add_observer(observer1)
    execution_status.value = Status.RUNNING
    assert observer1.status == Status.RUNNING

    observer2 = Observer()
    execution_status.add_observer(observer2)
    execution_status.value = Status.DONE
    assert observer1.status == Status.DONE
    assert observer2.status == Status.DONE

    execution_status.remove_observer(observer1)
    execution_status.value = Status.FAILED
    assert observer1.status == Status.DONE
    assert observer2.status == Status.FAILED


def test_pickling(execution_status, tmp_wd):
    """Verify pickling."""
    to_pickle(execution_status, "pickle_path")
    execution_status = from_pickle("pickle_path")
    assert execution_status.value == Status.DONE
