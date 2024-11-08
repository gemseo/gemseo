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
"""Base class for extendable execution sequence."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING

from gemseo.core._process_flow.execution_sequences.base_composite import (
    BaseCompositeExecSequence,
)
from gemseo.core._process_flow.execution_sequences.execution_sequence import (
    ExecutionSequence,
)
from gemseo.core.execution_status import ExecutionStatus

if TYPE_CHECKING:
    from gemseo.core._process_flow.execution_sequences.base import BaseExecutionSequence
    from gemseo.core.discipline import Discipline

_Status = ExecutionStatus.Status


class BaseExtendableExecSequence(BaseCompositeExecSequence):
    """A base class for composite execution sequence that are extendable."""

    def __init__(
        self,
        sequence: BaseExecutionSequence | Iterable[BaseExecutionSequence] = (),
    ) -> None:  # noqa:D107
        super().__init__()
        if sequence:
            self.extend(sequence)

    def extend(
        self,
        sequence: BaseExecutionSequence | Iterable[BaseExecutionSequence] | Discipline,
    ) -> None:
        """Extend the execution sequence with another sequence or discipline(s).

        Args:
            sequence: Either another execution sequence or one or several disciplines.

        Returns:
            The extended execution sequence.
        """
        # Avoids circular imports
        from gemseo.core.discipline import Discipline

        seq_class = sequence.__class__
        self_class = self.__class__
        if isinstance(sequence, Iterable):
            sequences = tuple(map(ExecutionSequence, sequence))
            uuid_to_disc = {s.uuid: s.process for s in sequences}
        elif isinstance(sequence, Discipline):
            atomic_sequence = ExecutionSequence(sequence)
            sequences = (atomic_sequence,)
            uuid_to_disc = {atomic_sequence.uuid: atomic_sequence.process}
        elif isinstance(sequence, ExecutionSequence):
            sequences = (sequence,)
            uuid_to_disc = {sequence.uuid: sequence}
        elif seq_class != self_class:
            sequences = (sequence,)
            uuid_to_disc = sequence.uuid_to_disc
        else:
            # seq_class == self_class
            sequences = sequence.sequences
            uuid_to_disc = sequence.uuid_to_disc

        self.sequences.extend(sequences)
        self.uuid_to_disc.update(uuid_to_disc)

        self._compute_disc_to_uuids()  # refresh disc_to_uuids
        for sequence in self.sequences:
            sequence.parent = self

    def _update_child_status(self, child: BaseExecutionSequence) -> None:
        """Manage status change of child execution sequences.

        Done status management is handled in subclasses.

        Args:
            child: The child execution sequence (contained in sequences)
                whose status has changed.
        """
        if child.status == _Status.FAILED:
            self.status = _Status.FAILED
        elif child.status == _Status.DONE:
            self._update_child_done_status(child)
        else:
            self.status = child.status

    @abstractmethod
    def _update_child_done_status(self, child: BaseExecutionSequence) -> None:
        """Handle done status of child execution sequences.

        Args:
            child: The child execution sequence (contained in sequences)
                whose status has changed.
        """
