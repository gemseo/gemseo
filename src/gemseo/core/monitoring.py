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
"""Monitoring mechanism to track |g| execution (update events)."""
from __future__ import annotations

from gemseo.core.scenario import Scenario
from gemseo.utils.singleton import SingleInstancePerAttributeId


class Monitoring(metaclass=SingleInstancePerAttributeId):
    """This class implements the observer pattern.

    It is a singleton, it is called by |g| core classes like MDODicipline whenever an
    event of interest like a status change occurs. Client objects register with
    add_observer and are notified whenever a discipline status change occurs.
    """

    def __init__(self, scenario: Scenario) -> None:
        """
        Args:
            scenario: The scenario to be monitored.
        """  # noqa: D205, D212, D415
        self._observers = []
        self.workflow = scenario.get_expected_workflow()
        self.workflow.set_observer(self)
        self.workflow.enable()

    def add_observer(self, observer) -> None:
        """Register an observer object interested in observable update events.

        Args:
            observer: The object to be notified.
        """
        if observer not in self._observers:
            self._observers.append(observer)

    def remove_observer(self, observer) -> None:
        """Unsubscribe the given observer.

        Args:
            observer: The observer to be removed.
        """
        if observer in self._observers:
            self._observers.remove(observer)

    def remove_all_observers(self) -> None:
        """Unsubscribe all observers."""
        self._observers = []

    def update(self, atom) -> None:
        """Notify the observers that the corresponding observable object is updated.

        Observers have to know what to retrieve from the observable object.

        Args:
            atom: The updated object.
        """
        for obs in self._observers:
            obs.update(atom)

    def get_statuses(self) -> dict[str, str]:
        """Get the statuses of all disciplines.

        Returns:
            These statuses associated with the atom ids.
        """
        return self.workflow.get_statuses()

    def __str__(self) -> str:
        return str(self.workflow)
