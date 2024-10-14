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
"""Loop execution sequence."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.core._process_flow.execution_sequences.base_composite import (
    BaseCompositeExecSequence,
)
from gemseo.core._process_flow.execution_sequences.execution_sequence import (
    ExecutionSequence,
)
from gemseo.core.execution_status import ExecutionStatus

if TYPE_CHECKING:
    from gemseo.core.discipline import Discipline

_Status = ExecutionStatus.Status


class LoopExecSequence(BaseCompositeExecSequence):
    """A loop with a controller discipline and an execution_sequence as iterate."""

    _PREFIX = "{"
    _SUFFIX = "}"

    def __init__(
        self,
        controller: Discipline | ExecutionSequence,
        sequence: BaseCompositeExecSequence,
    ) -> None:
        """
        Args:
            controller: A controller.
            sequence: A sequence.
        """  # noqa: D205, D212, D415
        if isinstance(controller, ExecutionSequence):
            atomic_sequence = controller
        else:
            atomic_sequence = ExecutionSequence(controller)
        super().__init__()
        self.sequences = [atomic_sequence, sequence]
        self.atom_controller = atomic_sequence
        self.atom_controller.parent = self
        self.iteration_sequence = sequence
        self.iteration_sequence.parent = self
        self.uuid_to_disc.update(sequence.uuid_to_disc)
        self.uuid_to_disc[self.atom_controller.uuid] = controller
        self._compute_disc_to_uuids()
        self.iteration_count = 0

    def _accept(self, visitor) -> None:
        visitor.visit_loop(self)

    def enable(self) -> None:
        super().enable()
        self.atom_controller.enable()
        self.iteration_count = 0

    def _update_child_status(self, child) -> None:
        """Update iteration successively regarding controller status.

        Count iterations regarding iteration_sequence status.

        Args:
            child: The child execution sequence in done state.
        """
        self.status = self.atom_controller.status
        if child == self.atom_controller:
            if self.status == _Status.RUNNING:
                if not self.iteration_sequence.is_enabled:
                    self.iteration_sequence.enable()
            elif self.status == _Status.DONE:
                self.disable()
                self.force_statuses(_Status.DONE)
        if child == self.iteration_sequence and child.status == _Status.DONE:
            self.iteration_count += 1
            self.iteration_sequence.enable()
        if child.status == _Status.FAILED:
            self.status = _Status.FAILED
