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
"""A sequential execution sequence."""

from __future__ import annotations

from gemseo.core._process_flow.execution_sequences.base_extendable import (
    BaseExtendableExecSequence,
)
from gemseo.core.execution_status import ExecutionStatus

_Status = ExecutionStatus.Status


class SequentialExecSequence(BaseExtendableExecSequence):
    """A sequential execution sequence of disciplines."""

    _PREFIX = "["
    _SUFFIX = "]"

    def __init__(self, sequence=()) -> None:  # noqa:D107
        super().__init__(sequence)
        self.exec_index = None

    def _accept(self, visitor) -> None:
        visitor.visit_serial(self)

    def enable(self) -> None:
        if not self.sequences:
            msg = "Serial execution is empty"
            raise ValueError(msg)
        super().enable()
        self.exec_index = 0
        self.sequences[self.exec_index].enable()

    def _update_child_done_status(self, child) -> None:
        """Update next child to given child execution sequence.

        Disable itself when all children done.

        Args:
            child: The child execution sequence in done state.
        """
        if child.status != _Status.DONE:
            return

        child.disable()
        self.exec_index += 1

        if self.exec_index < len(self.sequences):
            self.sequences[self.exec_index].enable()
        else:  # last seq done
            self.status = _Status.DONE
            self.disable()
