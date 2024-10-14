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
"""A system of disciplines."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.core.base_execution_status_observer import BaseExecutionStatusObserver
from gemseo.core.execution_status import ExecutionStatus

if TYPE_CHECKING:
    from gemseo.core.discipline.base_discipline import BaseDiscipline


class DisciplinesStatusObserver(BaseExecutionStatusObserver):
    """An execution status observer to propagate status."""

    __disciplines: tuple[BaseDiscipline, ...]
    """The disciplines."""

    def __init__(self, disciplines: tuple[BaseDiscipline, ...]):
        """
        Args:
            disciplines: The disciplines to be updated with an execution status.
        """  # noqa: D205, D212
        self.__disciplines = disciplines

    def update_status(self, execution_status: ExecutionStatus) -> None:  # noqa: D102
        if execution_status.value is ExecutionStatus.Status.PENDING:
            for discipline in self.__disciplines:
                discipline.execution_status.value = execution_status.value
