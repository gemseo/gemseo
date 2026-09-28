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
"""Base tracer for recording execution data during workflow observation."""

from __future__ import annotations

from os import getpid
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from gemseo.util._tracer._yaml import _NpyArrayStore
from gemseo.util._tracer._yaml import convert_to_yaml_data
from gemseo.util._tracer._yaml import dump_yaml_to_file
from gemseo.util._tracer.registry import TraceRegistry
from gemseo.util.metaclass import ABCGoogleDocstringInheritanceMeta
from gemseo.util.timer import Timer

if TYPE_CHECKING:
    from pathlib import Path

    from gemseo.util._workflow_observer.base_observer import BaseWorkflowObserver
    from gemseo.util.typing import MutableStrKeyMapping
    from gemseo.util.typing import StrKeyMapping


class BaseTracer(metaclass=ABCGoogleDocstringInheritanceMeta):
    """Base class for recording execution traces during workflow observation.

    Subclasses implement specific trace formats for different object types
    (disciplines, MDAs, optimizers, etc.), capturing input data, output data,
    and execution timing information.
    """

    __trace_file_name: ClassVar[str] = ".gemseo-trace.yml"
    """The name of the trace file."""

    __trace_arrays_directory_name: ClassVar[str] = ".gemseo-trace.arrays"
    """The name of the sibling directory holding the trace's referenced arrays."""

    _observed_object: Any
    """The object subject to tracing."""

    __object_id: str
    """The registry id of the observee, e.g. ``"MDAJacobi/0"``."""

    __timer: Timer
    """The timer for recording the observee call duration."""

    __timer_entered: bool
    """Whether `start` entered the timer.

    `start` may raise before entering the timer, e.g. `_get_start_trace`
    raising on a call argument with a raising `__str__`: `BaseDMProcessor.start`
    then logs the error and lets the observed call proceed untraced, still
    followed by a call to `end`. Without this flag, `end` would exit a timer
    that was never entered, computing a duration counted from the timer's
    construction, e.g. the process' whole uptime, instead of omitting the
    timing altogether.
    """

    __trace: MutableStrKeyMapping
    """The trace data accumulated during observation."""

    __array_store: _NpyArrayStore
    """The store for the large arrays met while converting the current cycle."""

    def __init__(
        self,
        observer: BaseWorkflowObserver,
        init_arguments: StrKeyMapping,
    ) -> None:
        """
        Args:
            observer: The workflow observer bound to this tracer.
            init_arguments: The normalized arguments used when instancing the
                observed object, by parameter name.
        """  # noqa: D205, D212
        self._observed_object = observer.object_
        self.__timer = Timer()
        self.__timer_entered = False
        self.__object_id = self.__resolve_object_id(init_arguments)
        self.__reset_cycle()

    def start(self, call_arguments: StrKeyMapping, directory_path: Path) -> None:
        """Start the tracing.

        The trace data is converted here, and not when the trace is written: a
        traced value is held by reference, so an observed call mutating one of
        its arguments in place, e.g. a discipline adding to its input array,
        would otherwise be traced with the value that argument had after the
        call. The arrays met during this conversion are written to
        `directory_path` immediately, for the same reason: holding a reference
        to a large array until `end` writes the trace would defeat that
        snapshot semantics just as much as holding the converted data would.

        Args:
            call_arguments: The normalized arguments of the observed call, by
                parameter name.
            directory_path: The path to the directory where the trace's
                arrays, if any, are written.
        """
        self.__trace.update(
            convert_to_yaml_data(
                self._get_start_trace(call_arguments), self.__array_store
            )
        )
        self.__array_store.write_arrays(directory_path)
        self.__timer.__enter__()
        self.__timer_entered = True

    def end(
        self,
        call_arguments: StrKeyMapping,
        returned_data: Any,
        directory_path: Path,
    ) -> None:
        """Finish tracing the observee call.

        The trace data is converted here, for the same reason as in `start`,
        and its arrays are written before the trace file itself: the latter is
        written atomically, see `dump_yaml_to_file`, so a reader that sees the
        YAML file already sees the arrays it references.

        Args:
            call_arguments: The normalized arguments of the observed call, by
                parameter name.
            returned_data: The data returned by the observed callable.
            directory_path: The path to the directory where the trace is written.
        """
        # `start` may have raised before entering the timer, see
        # `__timer_entered`'s documentation; exiting it now would then compute
        # a duration counted from the timer's construction instead of the
        # observed call's.
        if self.__timer_entered:
            self.__timer.__exit__(None, None, None)
        try:
            self.__trace.update(
                convert_to_yaml_data(
                    self._get_end_trace(call_arguments, returned_data),
                    self.__array_store,
                )
            )
            self.__array_store.write_arrays(directory_path)
            self.__write_trace(directory_path)
        finally:
            # Always seed the next cycle, even when this one could not be
            # traced: `BaseDMProcessor.end` logs the error instead of raising
            # it, so the data accumulated here would otherwise leak into the
            # trace of the next cycle.
            self.__reset_cycle()
            self.__timer_entered = False

    def _get_init_trace(self, init_arguments: StrKeyMapping) -> MutableStrKeyMapping:
        """Return the trace registry payload for the observee.

        The constructor arguments are normalized, so they are traced by
        parameter name, whatever the way they were passed. They are returned
        unconverted: `TraceRegistry.register` converts them itself, once it
        knows the entry name, since that name is what the arrays directory of
        a large constructor argument is named after.

        Args:
            init_arguments: The arguments used when the observee was instantiated.

        Returns:
            The registry payload: the observee's class documentation and its
            unconverted constructor arguments.
        """
        return {
            "documentation": self._observed_object.__class__.__doc__ or "",
            "init_arguments": init_arguments,
        }

    def _get_start_trace(
        self,
        call_arguments: StrKeyMapping,  # noqa: ARG002
    ) -> MutableStrKeyMapping:
        """Return the trace for the start method.

        Args:
            call_arguments: The normalized arguments of the observed call, by
                parameter name.

        Returns:
            The trace data, empty unless a subclass has something to record
            before the observed call runs.
        """
        return {}

    def _get_end_trace(
        self, call_arguments: StrKeyMapping, returned_data: Any
    ) -> MutableStrKeyMapping:
        """Return the trace for the end method.

        Args:
            call_arguments: The normalized arguments of the observed call, by
                parameter name.
            returned_data: The data returned by the observed callable.

        Returns:
            The trace data, omitting `start` and `duration` when `start`
            raised before entering the timer, see `__timer_entered`'s
            documentation.
        """
        if not self.__timer_entered:
            return {}

        timer = self.__timer
        # TODO: add the observed returned_data.
        return {
            "start": timer.entering_timestamp,
            "duration": timer.elapsed_time,
        }

    def __resolve_object_id(self, init_arguments: StrKeyMapping) -> str:
        """Resolve the registry object id of the observee.

        Reuses the id already stamped on the observee, if any -- e.g. by a
        sibling tracer created for the same instance, such as the execution
        and linearization tracers of one discipline, both built by
        `BaseWorkflowObserverDispatcher.__init__` -- otherwise registers the
        observee and stamps the new id onto it.

        The stamped id is only reused when it is not stale, see
        `__is_stale_id`; a stale one is discarded and the observee is
        registered again, as if it had never been.

        Args:
            init_arguments: The arguments used when the observee was instantiated.

        Returns:
            The registry object id of the observee.
        """
        observee = self._observed_object
        registry = TraceRegistry()
        object_id = getattr(observee, "_workflow_trace_id", None)
        if object_id is not None and not self.__is_stale_id(observee, registry):
            return object_id

        payload = self._get_init_trace(init_arguments)
        object_id = registry.register(
            type(observee).__name__,
            name=getattr(observee, "name", ""),
            documentation=payload["documentation"],
            init_arguments=payload["init_arguments"],
        )
        # This assignment requires an observee accepting new attributes, which holds
        # for all the observed gemseo classes. An observee defining ``__slots__``
        # without this attribute, or rejecting assignments in ``__setattr__``, would
        # raise an ``AttributeError`` here; silencing it would be worse than failing,
        # since the id could no longer be shared and the sibling tracers of a same
        # observee would each register a distinct id in the trace registry.
        observee._workflow_trace_id = object_id
        # Stamped alongside the id so a later call can tell a stale id apart,
        # see `__is_stale_id`. Plain ints, unlike the registry itself, which
        # holds a lock and is therefore unpicklable: an observee, e.g. a
        # discipline, may be pickled to send it to a worker process.
        observee._workflow_trace_pid = getpid()
        observee._workflow_trace_generation = registry.generation
        return object_id

    @staticmethod
    def __is_stale_id(observee: Any, registry: TraceRegistry) -> bool:
        """Return whether the id stamped on the observee is stale.

        The id is stale when the process that stamped it is the current one,
        and that process has since replaced the registry that stamped it:
        `Settings.__reset_directory_manager` evicts `TraceRegistry` from the
        `BaseMultiton` cache whenever the directory manager is (re-)enabled,
        which restarts its per-class counters from 0. Reusing an id stamped
        by a registry that no longer exists would otherwise collide with the
        id the new registry assigns to a distinct observee of the same
        class, e.g. two unrelated objects both resolving to
        ``"Sellar1/0"``.

        An id stamped in a different process, e.g. unpickled from an
        observee constructed in the parent process before being sent to a
        worker, is never stale: that worker either shares the parent's
        registry (a forked one, inheriting a snapshot of it) or rebuilds an
        unrelated one from scratch (see `TraceRegistry`'s class
        documentation), so comparing generations across that boundary would
        be meaningless; the id must still be reused as is, matching the
        registry's own handling of that case.

        Args:
            observee: The object subject to tracing.
            registry: The current trace registry.

        Returns:
            Whether the stamped id must be discarded and the observee
            registered again.
        """
        stamped_pid = getattr(observee, "_workflow_trace_pid", None)
        # The stamping process is compared to the current one directly: asking
        # `is_in_worker_process` would answer whether the *current* process is
        # a worker, whatever the pid passed, so in a worker process no id
        # could ever be stale and two observees of one class would collide on
        # one id after a registry reset.
        if stamped_pid is None or stamped_pid != getpid():
            return False
        return getattr(observee, "_workflow_trace_generation", None) != (
            registry.generation
        )

    def __get_trace_seed(self) -> MutableStrKeyMapping:
        """Return the per-cycle trace seed.

        Returns:
            The trace seed: the observee's registry object id and type.
        """
        return {
            "object_id": self.__object_id,
            "type": type(self._observed_object).__name__,
        }

    def __reset_cycle(self) -> None:
        """Reset the per-cycle trace data and its array store.

        Called once per call cycle: by `__init__`, then again at the end of
        every `end`, so that the store used by `start` never reuses a file
        name written by a previous cycle of the same tracer, see
        `_NpyArrayStore`.
        """
        self.__trace = self.__get_trace_seed()
        self.__array_store = _NpyArrayStore(self.__trace_arrays_directory_name)

    def __write_trace(self, directory_path: Path) -> None:
        """Write the trace file to the execution directory.

        The trace has already been converted by `start` and `end`, as well as
        the seed it starts from, which holds strings only. It is written
        atomically, as a registry entry is, see `dump_yaml_to_file`.

        Args:
            directory_path: The path to the directory where the trace is written.
        """
        dump_yaml_to_file(self.__trace, directory_path / self.__trace_file_name)
