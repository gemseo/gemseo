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
"""A base class to define monitored processes."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import ClassVar

from gemseo.core.execution_statistics import ExecutionStatistics
from gemseo.core.execution_status import ExecutionStatus
from gemseo.core.serializable import Serializable

if TYPE_CHECKING:
    from gemseo.core._process_flow.base_process_flow import BaseProcessFlow
    from gemseo.utils.string_tools import MultiLineString


class BaseMonitoredProcess(Serializable):
    """A base class to define monitored processes.

    A monitored process is an object
    with an execution method,
    an execution status
    and execution statistics,
    e.g. :class:`.Discipline`, :class:`.ProcessDiscipline` and :class:`.Scenario`.
    """

    name: str
    """The name of the process."""

    execution_statistics: ExecutionStatistics
    """The execution statistics of the process."""

    execution_status: ExecutionStatus
    """The execution status of the process."""

    # TODO: use Generic type.
    _process_flow_class: ClassVar[type[BaseProcessFlow]]
    """The class used to create the process flow."""

    def __init__(self, name: str) -> None:
        """
        Args:
            name: The name of the process.
                If empty, use the class name.
        """  # noqa: D205, D212, D415
        self.name = name or self.__class__.__name__
        self.execution_statistics = ExecutionStatistics(self.name)
        self.execution_status = ExecutionStatus(self.name)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return str(self._get_string_representation())

    def _repr_html_(self) -> str:
        return self._get_string_representation()._repr_html_()

    @abstractmethod
    def _get_string_representation(self) -> MultiLineString:
        """Return the string representation of the object.

        Returns:
            The string representation of the object.
        """

    def _execute_monitored(self) -> None:
        """Execute and monitor the internal business logic.

        This method handles the execution of :meth:`._execute` monitored with
        the execution status and statistics.
        It shall be called by :meth:`.execute`.
        """
        self.execution_status.handle(
            self.execution_status.Status.RUNNING,
            self.execution_statistics.record_execution,
            self._execute,
        )

    @abstractmethod
    def _execute(self) -> None:
        """Execute the internal business logic.

        This shall contain the actual processing specific to a discipline.
        It shall be called by :meth:`._execute_monitored``.
        """

    @abstractmethod
    def execute(self) -> None:
        """Execute the business logic.

        This is the main entry point to use a discipline.
        """

    def get_process_flow(self) -> BaseProcessFlow:
        """Return the process flow.

        Returns:
            The process flow.
        """
        return self._process_flow_class(self)
