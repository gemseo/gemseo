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
"""Directory managers for discipline execution and linearization."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from gemseo.util._directory_manager.processor.base import BaseDMProcessor
from gemseo.util._workflow_observer.discipline import (
    DisciplineExecutionWorkflowObserver,
)
from gemseo.util._workflow_observer.discipline import (
    DisciplineLinearizationWorkflowObserver,
)

if TYPE_CHECKING:
    from gemseo.util._workflow_observer.base_observer import BaseWorkflowObserver


class DisciplineExecutionDMProcessor(BaseDMProcessor):
    """Directory manager for discipline execution events.

    Creates and manages directories for discipline execution observations,
    with directory names indicating the discipline and 'execution' phase.
    """

    observer_class: ClassVar[type[BaseWorkflowObserver]] = (
        DisciplineExecutionWorkflowObserver
    )

    def __str__(self) -> str:
        return f"{self._observer._object}_execution"


class DisciplineLinearizationDMProcessor(BaseDMProcessor):
    """Directory manager for discipline linearization events.

    Creates and manages directories for discipline linearization observations,
    with directory names indicating the discipline and 'linearization' phase.
    """

    observer_class: ClassVar[type[BaseWorkflowObserver]] = (
        DisciplineLinearizationWorkflowObserver
    )

    def __str__(self) -> str:
        return f"{self._observer._object}_linearization"
