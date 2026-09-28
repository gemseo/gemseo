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

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Any
from typing import Final

from gemseo.util._tracer.base import BaseTracer
from gemseo.util._workflow_observer.base_processor import BaseProcessor

if TYPE_CHECKING:
    from pathlib import Path

    from gemseo.util._directory_manager.manager import DirectoryManager
    from gemseo.util._workflow_observer.base_observer import BaseWorkflowObserver
    from gemseo.util._workflow_observer.interface import CallSpec
    from gemseo.util.typing import StrKeyMapping

logger: Final[logging.Logger] = logging.getLogger(__name__)
"""The logger of this module."""


class _NoOpTracer(BaseTracer):
    """Tracer that records nothing.

    Used as the fallback tracer of an observee whose real tracer could not be
    constructed, e.g. because one of its constructor arguments has a raising
    `__str__`: `start` and `end` then do nothing, so the observee is left
    untraced for its whole lifetime instead of its constructor being aborted.
    """

    def __init__(  # noqa: D107
        self,
        observer: BaseWorkflowObserver,
        init_arguments: StrKeyMapping,
    ) -> None:
        pass

    def start(self, call_arguments: StrKeyMapping, directory_path: Path) -> None:  # noqa: D102
        pass

    def end(  # noqa: D102
        self,
        call_arguments: StrKeyMapping,
        returned_data: Any,
        directory_path: Path,
    ) -> None:
        pass


class BaseDMProcessor(BaseProcessor):
    """Base processor for managing execution directories during observation.

    Handles directory creation and cleanup for observed objects, delegating
    to the global `DirectoryManager` singleton for filesystem operations.
    A processor also owns a tracer that records the execution data of the
    observed object in the execution directory.
    """

    _observer: BaseWorkflowObserver
    """The workflow observer managing this directory."""

    _tracer: BaseTracer
    """The tracer recording the execution data of the observed object."""

    __dm: DirectoryManager
    """The directory manager"""

    __directory_path: Path
    """The path of the execution directory of the current observation cycle.

    Returned by `DirectoryManager.start_directory`, instead of being searched
    back from the observer when the tracer needs it: the manager maps a
    directory path to an observer, so the reverse lookup is a linear scan over
    every directory started since the beginning of the process.

    Set by `start`, the only way to reach `end`: `BaseWorkflowObserver.end`
    refuses to end an observation that has not started. It is nonetheless read
    inside the guarded block of `end`, so that the `AttributeError` of a
    hypothetical unstarted cycle is logged like any other tracing error.

    Keeping the path is not strictly equivalent to searching it back: the
    manager renames an existing directory when a homonymic one is created (see
    `DirectoryManager.__get_directory_path`), so a path kept here goes stale if
    that happens while this cycle is open. Which takes two threads starting a
    directory of the same name under one parent, since a single thread has its
    working directory inside the open directory, hence cannot compute the same
    path again. The trace of that cycle is then lost, and logged as such, while
    the directory itself is still ended, `end_directory` doing its own lookup.
    """

    def __init__(  # noqa: D107
        self,
        observer: BaseWorkflowObserver,
        init_arguments: StrKeyMapping,
    ) -> None:
        self._observer = observer
        # Avoid import cycle.
        from gemseo.util._directory_manager.manager import DirectoryManager

        # Constructing the manager first also injects the execution root
        # into the trace registry before the tracer below registers its
        # observee.
        self.__dm = DirectoryManager()
        try:
            self._tracer = self._tracer_class(observer, init_arguments)
        except Exception:
            # Tracing shall never change the outcome of the observed call, not
            # even its construction: an init argument with a raising `__str__`
            # would otherwise propagate from here through
            # `convert_to_yaml_data`, called while registering the observee in
            # the trace registry.
            # Symmetric to start()'s and end()'s protection: the error is
            # logged and the observee is left untraced for its whole
            # lifetime, since the failed init arguments are not kept around
            # to retry the registration later.
            # The class name is logged rather than the processor: `__str__` may
            # read state that the observee sets later, or delegate to an
            # observee whose own `__str__` raises, which is precisely the
            # failure being reported. The gemseo log handlers format the record
            # outside the guard of `logging.Handler.emit` (see
            # `MultiLineHandlerMixin.emit`), so such an error would propagate
            # out of this block and abort the construction.
            logger.exception(
                "The tracing of an instance of %s could not be constructed.",
                type(self).__name__,
            )
            self._tracer = _NoOpTracer(observer, init_arguments)

    @property
    @abstractmethod
    def _tracer_class(self) -> type[BaseTracer]:
        """The tracer class bound to the current processor."""

    def start(self, call_spec: CallSpec) -> None:
        self.__directory_path = self.__dm.start_directory(self._observer, str(self))
        try:
            self._tracer.start(call_spec.kwargs, self.__directory_path)
        except Exception:
            # Tracing shall never change the outcome of the observed call: the
            # error is logged and the call proceeds untraced. The execution
            # directory is left open, since the observed call is about to run
            # in it and end() closes it.
            # The processor is logged rather than the observed callable: it
            # names both the observee and its execution directory, e.g.
            # ``"Sellar1_execution"``.
            logger.exception("The tracing of %s could not be started.", self)
        except BaseException:
            # An exception that is not an error, e.g. a KeyboardInterrupt, is
            # not swallowed. Symmetric to end()'s protection: restore the
            # directory before propagating, otherwise the working directory
            # would stay inside the execution directory, breaking every later
            # execution.
            self.__dm.end_directory(self._observer)
            raise

    def end(self, call_spec: CallSpec, returned_data: Any) -> None:  # noqa: D102
        try:
            self._tracer.end(call_spec.kwargs, returned_data, self.__directory_path)
        except Exception:
            # Symmetric to start()'s protection: the observed call has already
            # returned, so a tracing error shall not turn it into a failure.
            logger.exception("The tracing of %s could not be ended.", self)
        finally:
            # Always end the directory, even when tracing fails: otherwise
            # the working directory would never be restored to the parent,
            # breaking every later execution.
            self.__dm.end_directory(self._observer)

    def __str__(self) -> str:
        return str(self._observer.object_)
