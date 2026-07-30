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
"""Base class for workflow observer processors."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Any

from gemseo.util.metaclass import ABCGoogleDocstringInheritanceMeta

if TYPE_CHECKING:
    from gemseo.util._workflow_observer.base_observer import BaseWorkflowObserver
    from gemseo.util._workflow_observer.interface import CallArguments
    from gemseo.util._workflow_observer.interface import CallSpec


class BaseProcessor(metaclass=ABCGoogleDocstringInheritanceMeta):
    """Base class for observer processors.

    A processor mirrors the lifecycle of an observer.
    It is used to implement the actual processing done while observing an object.

    A processor begins when calling the method `start`
    and finishes when calling the method `end`.
    """

    @property
    @abstractmethod
    def observer_class(self) -> type[BaseWorkflowObserver]:
        """The observer class bound to the current processor."""

    @abstractmethod
    def __init__(
        self,
        observer: BaseWorkflowObserver,
        init_arguments: CallArguments,
    ) -> None:
        """
        Args:
            observer: The observer bound to the current processor.
            init_arguments: The arguments used when instancing the observed object.
        """  # noqa: D205, D212

    @abstractmethod
    def start(self, call_spec: CallSpec) -> None:
        """Start the processor.

        Args:
            call_spec: The call specification of the observed method.
        """

    @abstractmethod
    def end(self, call_spec: CallSpec, returned_data: Any) -> None:
        """Finish the processor.

        Args:
            call_spec: The call specification of the observed method.
            returned_data: The data returned by the observed method.
        """
