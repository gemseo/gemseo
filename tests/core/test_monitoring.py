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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                         documentation
#        :author: Remi Lafage
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import unittest

from gemseo.core._process_flow.base_process_flow import BaseProcessFlow
from gemseo.core._process_flow.execution_sequences.sequential import (
    SequentialExecSequence,
)
from gemseo.core.execution_status import ExecutionStatus
from gemseo.core.monitoring import Monitoring
from gemseo.utils.discipline import DummyDiscipline


class ProcessWorkflow(BaseProcessFlow):
    def get_execution_flow(self):
        wkf = SequentialExecSequence()
        wkf.extend([self._node.disc1, self._node.disc2])
        return wkf


class FakeScenario:
    def __init__(self, disc1, disc2) -> None:
        self.disc1 = disc1
        self.disc2 = disc2

    def get_process_flow(self):
        return ProcessWorkflow(self)


class TestMonitoring(unittest.TestCase):
    def setUp(self) -> None:
        self.sc = FakeScenario(DummyDiscipline(), DummyDiscipline())
        self.monitor = Monitoring(self.sc)
        self.monitor.add_observer(self)
        self._statuses = self.monitor.get_statuses()
        self._updated_uuid = None

    def update(self, atom) -> None:
        self._statuses = self.monitor.get_statuses()
        self._updated_uuid = atom.uuid

    def _assert_update_status(self, disc, expected) -> None:
        disc.execution_status.value = expected
        assert expected == self._statuses[self._updated_uuid]

    def test_singleton(self) -> None:
        self.info = None

        class Observer2:
            def update(self, atom) -> None:
                self.info = atom

        observer2 = Observer2()

        monitor2 = Monitoring(self.sc)
        monitor2.add_observer(observer2)
        assert id(monitor2) == id(self.monitor)

    def test_status_update(self) -> None:
        self._assert_update_status(self.sc.disc1, ExecutionStatus.Status.RUNNING)
        self._assert_update_status(self.sc.disc1, ExecutionStatus.Status.DONE)

    def test_remove_observer(self) -> None:
        self.monitor.remove_observer(self)
        self.sc.disc1.execution_status.value = ExecutionStatus.Status.RUNNING
        assert None is self._updated_uuid  # no update received
        # check second remove works
        self.monitor.remove_observer(self)

    def test_remove_observers(self) -> None:
        self.monitor.remove_all_observers()
        self.sc.disc1.execution_status.value = ExecutionStatus.Status.RUNNING
        assert None is self._updated_uuid  # no update received
