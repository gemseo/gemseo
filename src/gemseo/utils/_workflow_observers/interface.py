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
"""Interface for workflow observer classes."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any

from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class CallArguments:
    """Arguments passed to a callable during invocation."""

    args: tuple[Any, ...]
    """The positional arguments."""

    kwargs: dict[str, Any]
    """The keyword arguments."""


@dataclass
class CallSpec(CallArguments):
    """Complete specification of a callable invocation.

    Extends CallArguments with the callable reference to provide complete information
    about a method call, including what was called and with what arguments.
    """

    callable_: Callable[..., Any]
    """The callable."""


class WorkflowObserverInterface(metaclass=ABCGoogleDocstringInheritanceMeta):
    """Interface for workflow observer implementations.

    A workflow observer tracks the lifecycle of object execution by notifying
    about start and end events of observed methods. Implementations should handle
    these events to perform custom actions like logging, monitoring, or state tracking.
    """

    @abstractmethod
    def __init__(
        self,
        object_: object,
        init_arguments: CallArguments,
    ) -> None:
        """
        Args:
            object_: The object to observe.
            init_arguments: The arguments used when instantiating the object to observe.
        """  # noqa: D205, D212

    @abstractmethod
    def start(self, call_spec: CallSpec) -> None:
        """Start the observation.

        Args:
            call_spec: The call specification of the method to observe.
        """

    @abstractmethod
    def end(self, call_spec: CallSpec, returned_data: Any) -> None:
        """Finish the observation.

        Args:
            call_spec: The call specification of the method to observe.
            returned_data: The data returned by the method to observe.
        """
