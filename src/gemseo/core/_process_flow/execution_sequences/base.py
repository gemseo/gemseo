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
"""A base class for discipline execution sequences."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING
from uuid import uuid4

from gemseo.core.execution_status import ExecutionStatus
from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta

if TYPE_CHECKING:
    from gemseo.core._base_monitored_process import BaseMonitoredProcess
    from gemseo.core.base_execution_status_observer import BaseExecutionStatusObserver

_Status = ExecutionStatus.Status


class BaseExecutionSequence(metaclass=ABCGoogleDocstringInheritanceMeta):
    """A base class for discipline execution sequences.

    The execution sequence structure is introduced to reflect the main process flow
    implicitly executed by |g| regarding the given scenario/formulation executed. That
    structure allows to identify single executions of a same discipline that may be run
    several times at various stages in the given scenario/formulation.
    """

    _PREFIX = "["
    _SUFFIX = "]"

    uuid: str
    """The unique identifier of the sequence."""

    uuid_to_disc: dict[str, BaseMonitoredProcess]
    """The map from unique identifier to processes."""

    disc_to_uuids: dict[BaseMonitoredProcess, str]
    """The map from processes to unique identifier."""

    status: _Status
    """The execution status."""

    __is_enabled: bool
    """Whether the sequence is enabled."""

    _parent: BaseExecutionSequence | None
    """The parent of the current sequence."""

    def __init__(self) -> None:  # noqa: D107
        self.uuid = str(uuid4())
        self.uuid_to_disc = {}
        self.disc_to_uuids = {}
        self.status = _Status.DONE
        self.__is_enabled = False
        self._parent = None

    @abstractmethod
    def accept(self, visitor):
        """Accept a visitor object.

        See the Visitor pattern.

        Args:
            visitor: A visitor object.
        """

    @abstractmethod
    def set_observer(self, obs: BaseExecutionStatusObserver):
        """Register an observer.

        This observer is intended to be notified via its :meth:`update` method
        each time an underlying discipline changes its status.

        Returns:
            The disciplines.
        """

    @property
    def parent(self):
        """The execution sequence containing the current one.

        Raises:
            RuntimeError: When the current execution sequence is not a child
                of the given parent execution sequence.
        """
        return self._parent

    @parent.setter
    def parent(self, parent) -> None:
        if self not in parent.sequences:
            msg = f"parent {parent} does not include child {self}"
            raise RuntimeError(msg)
        self._parent = parent

    @property
    def is_enabled(self) -> bool:
        """Whether the sequence is enabled."""
        return self.__is_enabled

    # TODO: API: use a property setter for is_enabled.
    def enable(self) -> None:
        """Enable the execution sequence."""
        self.status = _Status.DONE
        self.__is_enabled = True

    def disable(self) -> None:
        """Disable the execution sequence."""
        self.__is_enabled = False

    def _compute_disc_to_uuids(self) -> None:
        """Update discipline to uuids mapping from uuids to discipline mapping.

        Notes:
            A discipline might correspond to several AtomicExecutionSequence hence
            might correspond to several uuids.
        """
        self.disc_to_uuids = {}
        for key, value in self.uuid_to_disc.items():
            self.disc_to_uuids.setdefault(value, []).append(key)
