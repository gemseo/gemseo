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
"""Base class for observers dispatchers."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from gemseo.utils._workflow_observers.interface import WorkflowObserverInterface

if TYPE_CHECKING:
    from gemseo.utils._workflow_observers.base_observer import BaseWorkflowObserver
    from gemseo.utils._workflow_observers.base_observer import ObservationSpec
    from gemseo.utils._workflow_observers.interface import CallArguments
    from gemseo.utils._workflow_observers.interface import CallSpec


class BaseWorkflowObserverDispatcher(WorkflowObserverInterface):
    """Base class for dispatcher observers that delegate to method-specific observers.

    Some objects require multiple observers depending on which method is called.
    This class acts as a facade, routing observation events to the appropriate
    method-specific observer based on the callable's name. The dispatcher itself
    does not perform observation; it delegates to its child observers.

    This is useful for objects with distinct lifecycle methods (e.g., discipline
    execution vs. linearization) that need separate observation strategies.
    """

    _spec: ClassVar[ObservationSpec]
    """The specifications for the base classes and methods to observe."""

    _method_name_to_observer_class: ClassVar[dict[str, type[BaseWorkflowObserver]]]
    """The mapping from method name to observer class."""

    __method_name_to_observer: dict[str, BaseWorkflowObserver]
    """The mapping from method name to observer."""

    def __init__(  # noqa: D107
        self,
        object_: object,
        init_arguments: CallArguments,
    ) -> None:
        self.__method_name_to_observer = {
            method_name: observer_class(object_, init_arguments)
            for method_name, observer_class in self._method_name_to_observer_class.items()  # noqa: E501
        }

    def start(self, call_spec: CallSpec) -> None:  # noqa: D102
        self.__method_name_to_observer[call_spec.callable_.__name__].start(call_spec)

    def end(self, call_spec: CallSpec, returned_data: Any) -> None:  # noqa: D102
        self.__method_name_to_observer[call_spec.callable_.__name__].end(
            call_spec, returned_data
        )
