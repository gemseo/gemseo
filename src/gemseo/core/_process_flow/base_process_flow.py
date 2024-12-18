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
"""Base class for process flows."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.core._process_flow.base_flow import BaseFlow
from gemseo.core._process_flow.execution_sequences.parallel import ParallelExecSequence
from gemseo.core._process_flow.execution_sequences.sequential import (
    SequentialExecSequence,
)

if TYPE_CHECKING:
    from gemseo.core._process_flow.execution_sequences.base import BaseExecutionSequence
    from gemseo.core.discipline.base_discipline import BaseDiscipline


class BaseProcessFlow(BaseFlow):
    """A base class for the process flow of a system of disciplines.

    A process flow is composed of a data flow and an execution flow.
    """

    is_parallel: bool = False
    """Whether the execution sequence is parallel or sequential."""

    def get_execution_flow(self) -> BaseExecutionSequence:
        """Return the execution flow.

        Returns:
            The execution flow.
        """
        sequence = (
            ParallelExecSequence() if self.is_parallel else SequentialExecSequence()
        )
        for discipline in self._node._disciplines:
            sequence.extend(discipline.get_process_flow().get_execution_flow())
        return sequence

    def get_disciplines_in_data_flow(self) -> list[BaseDiscipline]:
        """Return the disciplines that must be shown as blocks in the XDSM.

        Returns:
            The disciplines.
        """
        all_disciplines = []
        for disc in self._node._disciplines:
            disciplines = disc.get_process_flow().get_disciplines_in_data_flow()
            if not disciplines:
                disciplines = [disc]
            all_disciplines.extend(disciplines)
        return all_disciplines
