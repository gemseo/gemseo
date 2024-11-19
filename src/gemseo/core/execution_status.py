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
"""Execution status of processes."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import ClassVar

from strenum import StrEnum

from gemseo.core.serializable import Serializable

if TYPE_CHECKING:
    from gemseo.core.base_execution_status_observer import BaseExecutionStatusObserver

# TODO: use BaseExecutionStatusObserver.
# TODO: API: add more precise status rules?


class ExecutionStatus(Serializable):
    """Execution status of a process.

    The status aims at monitoring a process and give the user a simplified view on
    the state of the processes.

    The possible statuses are defined in :attr:`.Status`.
    The status rules are:
    - the initial status is ``DONE``,
    - the status ``RUNNING`` or ``LINEARIZING`` can only be set when the current one is
        ``DONE``,
    - the status ``DONE`` can only be set when the current one is ``RUNNING``.

    Helper methods should be used to handle the statuses when running or linearizing
    a process: :meth:`.run` and :meth:`linearize`.

    Observers can be attached and are notified when the value of the status is changed.
    The observers are not restored after pickling.
    """

    class Status(StrEnum):
        """The statuses."""

        RUNNING = "RUNNING"
        LINEARIZING = "LINEARIZING"
        FAILED = "FAILED"
        DONE = "DONE"

    _ATTR_NOT_TO_SERIALIZE: ClassVar[set[str]] = {"__observers"}

    __process_name: str
    """The name of the process that has an execution status."""

    __status: Status
    """The status."""

    __observers: set[BaseExecutionStatusObserver]
    """The observers."""

    def __init__(self, process_name: str) -> None:
        """
        Args:
            process_name: The name of the process.
        """  # noqa: D205, D212
        self.__process_name = process_name
        self.__status = self.Status.DONE
        self._init_shared_memory_attrs_before()

    def __check_can_be_running(self, status: Status) -> bool:
        """Return whether the status can be set to ``RUNNING``.

        Args:
            status: A status.

        Returns:
            Whether the status can be set to ``RUNNING``.
        """
        return status == self.Status.RUNNING and self.value != self.Status.DONE

    def __check_can_be_linearizing(self, status: Status) -> bool:
        """Return whether the status can be set to ``LINEARIZING``.

        Args:
            status: A status.

        Returns:
            Whether the status can be set to ``LINEARIZING``.
        """
        return status == self.Status.LINEARIZING and self.value != self.Status.DONE

    @property
    def value(self) -> Status:
        """The status of the process."""
        return self.__status

    @value.setter
    def value(self, status: Status) -> None:
        if self.__check_can_be_running(status) or self.__check_can_be_linearizing(
            status
        ):
            msg = (
                f"{self.__process_name} cannot be set to status {status} "
                f"while in status {self.value}."
            )
            raise ValueError(msg)

        # Cast the argument so that the enum class raise an explicit error.
        self.Status(status)
        self.__status = status
        self.__notify_observers()

    def handle(
        self,
        status: Status,
        function: Callable[[Any], None],
        *args: Any,
    ) -> None:
        """Handle a status while executing a function.

        On exception, the status is set to ``FAILED``, otherwise is set to ``DONE``.

        Args:
            status: The status to be set before execution.
            function: The function to be called.
            *args: The argument to be passed for calling the function.
        """
        self.value = status
        try:
            function(*args)
        except Exception:
            self.value = self.Status.FAILED
            raise
        self.value = self.Status.DONE

    def add_observer(self, observer: BaseExecutionStatusObserver) -> None:
        """Add an observer.

        Args:
            observer: The observer to add.
        """
        self.__observers.add(observer)

    def remove_observer(self, observer: BaseExecutionStatusObserver) -> None:
        """Remove an observer.

        Args:
            observer: The observer to remove.
        """
        self.__observers.discard(observer)

    def __notify_observers(self) -> None:
        """Notify the observers."""
        # Iterate on a copy because some ExecutionSequence observers can remove
        # observers.
        for observer in tuple(self.__observers):
            observer.update_status(self)

    def _init_shared_memory_attrs_before(self) -> None:
        self.__observers = set()
