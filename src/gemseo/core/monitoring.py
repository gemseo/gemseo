# -*- coding: utf-8 -*-
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
"""
Monitoring mechanism to track |g| execution (update events)
***********************************************************
"""
from __future__ import division, unicode_literals

from six import with_metaclass

from gemseo.utils.singleton import SingleInstancePerAttributeId


class Monitoring(with_metaclass(SingleInstancePerAttributeId, object)):
    """This class implements the observer pattern.

    It is a singleton, it is called by |g| core classes like MDODicipline whenever an
    event of interest like a status change occurs. Client objects register with
    add_observer and are notified whenever a discipline status change occurs.
    """

    def __init__(self, scenario):
        """Initializes the monitoring.

        :param scenario: the scenario to be monitored.
        :type scenario: MDOScenario
        """
        self._observers = []
        self.workflow = scenario.get_expected_workflow()
        self.workflow.set_observer(self)
        self.workflow.enable()

    def add_observer(self, observer):
        """Registers an observer object interested in observable update events.

        :param observer: object to be notified
        """
        if observer not in self._observers:
            self._observers.append(observer)

    def remove_observer(self, observer):
        """Unsubscribes the given observer.

        :param observer: observer to be removed
        """
        if observer in self._observers:
            self._observers.remove(observer)

    def remove_all_observers(self):
        """Unsubscribes all observers."""
        self._observers = []

    def update(self, atom):
        """Notifies observers that the corresponding observable object is updated.
        Observers have to know what to retrieve from the observable object.

        :param atom: updated object
        """
        for obs in self._observers:
            obs.update(atom)

    def get_statuses(self):
        """Gets the statuses of all disciplines.

        :returns: a dictionary of all statuses, keys are the atom ids
        :rtype: dict
        """
        return self.workflow.get_state_dict()

    def __str__(self):
        """Returns the string representation of the workflow.

        :returns: string representation of the workflow
        :rtype: str
        """
        return str(self.workflow)
