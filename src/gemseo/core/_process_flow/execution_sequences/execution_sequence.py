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
"""A discipline execution sequence."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Protocol

from gemseo.core._process_flow.execution_sequences.base import BaseExecutionSequence
from gemseo.core.execution_status import ExecutionStatus

if TYPE_CHECKING:
    from gemseo.core._base_monitored_process import BaseMonitoredProcess
    from gemseo.core.base_execution_status_observer import BaseExecutionStatusObserver

_Status = ExecutionStatus.Status


class Visitor(Protocol):
    def visit_atomic(self, sequence: ExecutionSequence) -> None:
        pass


class ExecutionSequence(BaseExecutionSequence):
    """The execution sequence of a process."""

    _observer: BaseExecutionStatusObserver | None
    """An observer following the observer design pattern.

    Defined by :meth:`.set_observer`.
    """

    def __init__(self, process: BaseMonitoredProcess) -> None:
        """
        Args:
            process: A process.
        """  # noqa: D205, D212, D415
        super().__init__()
        self.process = process
        self.uuid_to_disc = {self.uuid: process}
        self.disc_to_uuids = {process: [self.uuid]}
        self._observer = None

    def __str__(self) -> str:
        return f"{self.process.name}({self.status})"

    def __repr__(self) -> str:
        return f"{self.process.name}({self.status}, {self.uuid})"

    def accept(self, visitor: Visitor) -> None:
        visitor.visit_atomic(self)

    def set_observer(self, obs: BaseExecutionStatusObserver) -> None:
        self._observer = obs

    def enable(self) -> None:
        super().enable()
        self.process.execution_status.add_observer(self)

    def disable(self) -> None:
        super().disable()
        self.process.execution_status.remove_observer(self)

    def get_statuses(self) -> dict[str, ExecutionStatus.Status]:
        """Return the statuses mapping atom uuid to status.

        Returns:
            The statuses mapping atom uuid to status.
        """
        return {self.uuid: self.status}

    def update_status(self, status: ExecutionStatus) -> None:
        """Update the status.

        Reflect the status then notifies the parent and the observer if any.
        Notes: update_status if discipline status change actually
        compared to current, otherwise do nothing.

        Args:
            status: The new status.
        """
        if self.is_enabled and self.status != status.value:
            self.status = status.value or _Status.DONE
            if self.status in {_Status.DONE, _Status.FAILED}:
                self.disable()
            if self._parent:
                self._parent.update_child_status(self)
            if self._observer:
                self._observer.update(self)

    def force_statuses(self, status: ExecutionStatus) -> None:
        """Force the self status and the status of subsequences.

        This is done without notifying the
        parent (as the force_status is called by a parent), but notify the observer is
        status changed.

        Args:
            status: The new status.
        """
        old_status = self.status
        self.status = status
        if old_status != status and self._observer:
            self._observer.update(self)
