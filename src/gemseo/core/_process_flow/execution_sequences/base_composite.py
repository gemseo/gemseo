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
"""A base class for composite execution sequence."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from gemseo.core._process_flow.execution_sequences.base import BaseExecutionSequence

if TYPE_CHECKING:
    from gemseo.core._process_flow.execution_sequences.execution_sequence import Visitor
    from gemseo.core.base_execution_status_observer import BaseExecutionStatusObserver
    from gemseo.core.discipline import Discipline
    from gemseo.core.execution_status import ExecutionStatus


class BaseCompositeExecSequence(BaseExecutionSequence):
    """A base class for execution sequence made of other execution sequences."""

    _PREFIX = "'"
    _SUFFIX = "'"

    sequences: list[BaseExecutionSequence]
    """The inner execution sequences."""

    disciplines: list[Discipline]
    """The disciplines."""

    def __init__(self) -> None:  # noqa:D107
        super().__init__()
        self.sequences = []
        self.disciplines = []

    def __str__(self) -> str:
        return self._PREFIX + ", ".join(map(str, self.sequences)) + self._SUFFIX

    def accept(self, visitor: Visitor) -> None:
        self._accept(visitor)
        for sequence in self.sequences:
            sequence.accept(visitor)

    @abstractmethod
    def _accept(self, visitor: Visitor) -> None:
        """Accept a visitor object.

        Args:
            visitor: An object implementing the `visit_serial()` method.
        """

    def set_observer(self, obs: BaseExecutionStatusObserver) -> None:
        for sequence in self.sequences:
            sequence.set_observer(obs)

    @BaseExecutionSequence.is_enabled.setter
    def is_enabled(self, enable: bool) -> None:
        super(__class__, self.__class__).is_enabled.fset(self, enable)
        if not enable:
            for sequence in self.sequences:
                sequence.is_enabled = False

    # TODO: factorize with ExecutionSequence (mixin?)
    def force_statuses(self, status: ExecutionStatus) -> None:
        """Set the self status and the status of subsequences.

        The change of status is not notified to the parent
        (as the `force_statuses` is called by a parent),
        but to the observer.

        Args:
            status: The new status.
        """
        self.status = status
        for sequence in self.sequences:
            sequence.force_statuses(status)

    # TODO: factorize with ExecutionSequence (mixin?)
    def get_statuses(self) -> dict[str, ExecutionStatus.Status]:
        """Return the statuses mapping atom uuid to status.

        Returns:
            The statuses mapping atom uuid to status.
        """
        uuid_to_status = {}
        for sequence in self.sequences:
            uuid_to_status.update(sequence.get_statuses())
        return uuid_to_status

    def update_child_status(self, child: BaseExecutionSequence) -> None:
        """Manage status change of child execution sequences.

        Propagates status change
        to the parent (containing execution sequence).

        Args:
            child: The child execution sequence (contained in sequences)
                whose status has changed.
        """
        old_status = self.status
        self._update_child_status(child)
        if self._parent and self.status != old_status:
            self._parent.update_child_status(self)

    @abstractmethod
    def _update_child_status(self, child: BaseExecutionSequence):
        """Handle child execution change.

        Args:
            child: the child execution sequence (contained in sequences)
                whose status has changed.
        """
