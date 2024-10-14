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

from __future__ import annotations

from typing import TYPE_CHECKING

from docstring_inheritance import GoogleDocstringInheritanceMeta

from gemseo.core._process_flow.execution_sequences.execution_sequence import (
    ExecutionSequence,
)

if TYPE_CHECKING:
    from gemseo import Discipline
    from gemseo.core._base_monitored_process import BaseMonitoredProcess
    from gemseo.core._process_flow.execution_sequences.base import BaseExecutionSequence
    from gemseo.core.discipline.base_discipline import BaseDiscipline


class BaseFlow(metaclass=GoogleDocstringInheritanceMeta):
    """A base class for the process flow of a discipline.

    A process flow is composed of a data flow and an execution flow.
    """

    _node: BaseMonitoredProcess
    """The object that is a node of the process flow, containing a sub-process flow."""

    def __init__(self, node: BaseMonitoredProcess) -> None:
        """
        Args:
            node: The object that is a node of the process flow,
                containing a sub-process flow.
        """  # noqa: D205, D212
        self._node = node

    def get_execution_flow(self) -> BaseExecutionSequence:
        """Return the execution flow.

        Returns:
            The execution flow.
        """
        return ExecutionSequence(self._node)

    def get_data_flow(
        self,
    ) -> list[tuple[Discipline, Discipline, list[str]]]:
        """Return the data flow.

        Returns:
            The data exchange arcs of the form
            ``(first_discipline, second_discipline, coupling_names)``.
        """
        return []

    def get_disciplines_in_data_flow(self) -> list[BaseDiscipline]:
        """Return the disciplines that must be shown as blocks in the XDSM.

        Returns:
            The disciplines.
        """
        return []
