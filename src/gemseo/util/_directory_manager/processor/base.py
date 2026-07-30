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
"""Base processor for directory management during workflow observation."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from gemseo.util._workflow_observer.base_processor import BaseProcessor

if TYPE_CHECKING:
    from gemseo.util._directory_manager.manager import DirectoryManager
    from gemseo.util._workflow_observer.base_observer import BaseWorkflowObserver
    from gemseo.util._workflow_observer.interface import CallArguments
    from gemseo.util._workflow_observer.interface import CallSpec


class BaseDMProcessor(BaseProcessor):
    """Base processor for managing execution directories during observation.

    Handles directory creation and cleanup for observed objects, delegating
    to the global `DirectoryManager` singleton for filesystem operations.
    """

    _observer: BaseWorkflowObserver
    """The workflow observer managing this directory."""

    __dm: DirectoryManager
    """The directory manager"""

    def __init__(  # noqa: D107
        self,
        observer: BaseWorkflowObserver,
        init_arguments: CallArguments,
    ) -> None:
        self._observer = observer
        # Avoid import cycle.
        from gemseo.util._directory_manager.manager import DirectoryManager

        self.__dm = DirectoryManager()

    def start(self, call_spec: CallSpec) -> None:
        self.__dm.start_directory(self._observer, str(self))

    def end(self, call_spec: CallSpec, returned_data: Any) -> None:  # noqa: D102
        self.__dm.end_directory(self._observer)

    def __str__(self) -> str:
        return str(self._observer._object)
