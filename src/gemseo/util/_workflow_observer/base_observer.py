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
"""Base observer, observation specification, and injectable observer protocol."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Final
from typing import Protocol
from typing import runtime_checkable

from gemseo.util._directory_manager.processor.factory import DM_PROCESSOR_FACTORY
from gemseo.util._workflow_observer.interface import WorkflowObserverInterface
from gemseo.util._workflow_observer.tree import ObserverTree

if TYPE_CHECKING:
    from gemseo.util._directory_manager.processor.factory import DMProcessorFactory
    from gemseo.util._workflow_observer.base_processor import BaseProcessor
    from gemseo.util._workflow_observer.interface import CallArguments
    from gemseo.util._workflow_observer.interface import CallSpec


@dataclass
class Status:
    """Status tracking for workflow observation lifecycle."""

    is_started: bool = False
    """Whether the observation has started and not yet finished."""


class BaseWorkflowObserver(WorkflowObserverInterface):
    """Base class for workflow observers.

    A workflow observer is used to observe the execution of an object
    in the gemseo workflow.
    An observation begins when calling the method `start`
    and finishes when calling the method `end`.
    A workflow observer can be nested in another workflow observer,
    the former being the child of the latter.
    The gathering of workflow observers parent-child relationships
    is done in the `ObserverTree`, this is a global object.
    A workflow observer belongs to the `ObserverTree` as long as it is active,
    i.e. between the calls to the methods `start` and `end`.

    !!! note
    Some observations can start in a method but ends in another method.

    The actual processing of the observed object is delegated to a processor,
    be it for the management of execution directories, the storage of data lineage, etc.
    For the time being, the only available processor is the directory manager.
    The goal is to allow to use more than one processor.
    """

    __observer_tree: Final[ObserverTree] = ObserverTree()
    """The global tree of observers."""

    __processor_factory: Final[DMProcessorFactory] = DM_PROCESSOR_FACTORY
    """The observation processor factory."""

    _object: object
    """The observed object."""

    _status: Status
    """The status for the observation of a workflow."""

    _processor: BaseProcessor
    """The object that does the actual processing of the observed object."""

    # __stream_handler: StreamHandler
    # """The stream handler for storing logging messages."""

    def __init__(  # noqa: D107
        self,
        object_: object,
        init_arguments: CallArguments,
    ) -> None:
        self._object = object_
        self._status = Status()
        self._processor = self.__processor_factory.create(self, init_arguments)
        # self.__stream_handler = StreamHandler(StringIO())

    def start(self, call_spec: CallSpec) -> None:  # noqa: D102
        if self._status.is_started:
            msg = "Cannot start an already started observer."
            raise RuntimeError(msg)
        self._processor.start(call_spec)
        self._status.is_started = True
        self.__observer_tree.put(self)
        # self.__add_logging_handler()

    def end(self, call_spec: CallSpec, returned_data: Any) -> None:  # noqa: D102
        if not self._status.is_started:
            msg = "Cannot finish an observer that has not been started."
            raise RuntimeError(msg)
        self._status.is_started = False
        try:
            self._processor.end(call_spec, returned_data)
        finally:
            # Always remove the observer from the tree, even when the processor
            # fails: a leftover entry would corrupt the parent-child bookkeeping
            # for the rest of the process.
            # TODO: Pass the current observer and check it is the one removed?
            self.__observer_tree.pop()
        # self.__remove_logging_handler()

    # TODO: Fix logging handling

    # def __add_logging_handler(self) -> None:
    #     """Add a logging handler to for logging in the observer directory."""
    #     # Logging message may come from any class ancestors,
    #     # thus all module loggers are instrumented.
    #     for class_ in self._object.__class__.__mro__:
    #         module_logger = getLogger(class_.__module__)
    #         module_logger.addHandler(self.__stream_handler)

    # def __remove_logging_handler(self) -> None:
    #     """Reset the handler and remove it from the loggers.
    #
    #     Otherwise,
    #     the module logger would send logging messages for the handler
    #     even if the related object is not being currently used.
    #     """
    #     stream_handler = self.__stream_handler
    #     for class_ in self._object.__class__.__mro__:
    #         module_logger = getLogger(class_.__module__)
    #         for handler in module_logger.handlers[:]:
    #             if handler == stream_handler:
    #                 module_logger.removeHandler(handler)
    #     # Reset the string.
    #     stream_handler.stream = StringIO()

    # def get_logging_messages(self) -> str:
    #     """Return the logging messages emitted since last start."""
    #     return self.__stream_handler.stream.getvalue()


@runtime_checkable
class InjectableObserver(Protocol):
    """Protocol for observer classes that can be injected into target classes."""

    _spec: ClassVar[ObservationSpec]


@dataclass
class ObservationSpec:
    """Specification defining which classes and methods to observe."""

    base_class: str
    """The fully qualified name of the base class to observe."""

    excluded_sub_classes: set[str] = field(default_factory=set)
    """Set of fully qualified subclass names to exclude from observation."""

    method_names_for_start: set[str] = field(default_factory=set)
    """Method names where only the start event is observed."""

    method_names_for_finish: set[str] = field(default_factory=set)
    """Method names where only the finish event is observed."""

    method_names_for_both: set[str] = field(default_factory=set)
    """Method names where both start and finish events are observed."""
