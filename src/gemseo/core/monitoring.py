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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Remi Lafage
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Monitoring mechanism to track GEMSEO execution (update events)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Protocol

from gemseo.utils.singleton import SingleInstancePerAttributeId

if TYPE_CHECKING:
    from gemseo.core._process_flow.execution_sequences.base import BaseExecutionSequence
    from gemseo.core.execution_status import ExecutionStatus

    class Observer(Protocol):
        """API of an observer."""

        def update(self, obj: Any) -> None:
            """Update an observer.

            Args:
                obj: The object to update from.
            """


class Monitoring(metaclass=SingleInstancePerAttributeId):
    """This class implements the observer pattern.

    It is a singleton.
    It is called by GEMSEO core classes like Discipline
    whenever an event of interest like a status change occurs.
    Client objects register with `add_observer
    and are notified whenever a status change occurs.
    """

    __observers: list[Observer]
    """The observers."""

    __execution_sequence: BaseExecutionSequence
    """The execution sequence."""

    def __init__(self, execution_sequence: BaseExecutionSequence) -> None:
        """
        Args:
            execution_sequence: The execution sequence structure.
        """  # noqa: D205, D212, D415
        self.__observers = []
        self.__execution_sequence = execution_sequence
        self.__execution_sequence.set_observer(self)
        self.__execution_sequence.is_enabled = True

    def add_observer(self, observer: Observer) -> None:
        """Register an observer object interested in observable update events.

        Args:
            observer: The object to be notified.
        """
        if observer not in self.__observers:
            self.__observers.append(observer)

    def remove_observer(self, observer: Observer) -> None:
        """Unsubscribe the given observer.

        Args:
            observer: The observer to be removed.
        """
        if observer in self.__observers:
            self.__observers.remove(observer)

    def remove_all_observers(self) -> None:
        """Unsubscribe all observers."""
        self.__observers.clear()

    def update(self, atom: Any) -> None:
        """Notify the observers that the corresponding observable object is updated.

        Observers have to know what to retrieve from the observable object.

        Args:
            atom: The updated object.
        """
        for obs in self.__observers:
            obs.update(atom)

    def get_statuses(self) -> dict[str, ExecutionStatus.Status]:
        """Get the statuses of all disciplines.

        Returns:
            These statuses associated with the atom ids.
        """
        return self.__execution_sequence.get_statuses()

    def __str__(self) -> str:
        return str(self.__execution_sequence)
