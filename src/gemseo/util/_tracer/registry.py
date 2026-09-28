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
"""Registry of the static per-instance trace data of observed objects."""

from __future__ import annotations

import logging
from dataclasses import asdict
from dataclasses import dataclass
from itertools import count
from os import getpid
from threading import Lock
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Final

from gemseo.util._filename_sanitizer import secure_filename
from gemseo.util._tracer._yaml import _NpyArrayStore
from gemseo.util._tracer._yaml import convert_to_yaml_data
from gemseo.util._tracer._yaml import dump_yaml_to_file
from gemseo.util._worker_context import is_in_worker_process
from gemseo.util.base_multiton import BaseMultiton

if TYPE_CHECKING:
    from pathlib import Path

logger: Final[logging.Logger] = logging.getLogger(__name__)
"""The logger of this module."""


@dataclass(frozen=True)
class _TraceRegistryEntry:
    """The schema of a trace registry entry."""

    object_id: str
    """The registry id of the observee, e.g. ``"MDAJacobi/0"``."""

    type: str
    """The name of the observee's class."""

    name: str
    """The name of the observee, empty if it has none."""

    documentation: str
    """The observee's class documentation."""

    init_arguments: Any
    """The converted arguments used to instantiate the observee."""


class TraceRegistry(metaclass=BaseMultiton):
    """Registry of the static trace data of each observed object.

    An observed object is registered once, when its tracer is constructed, as
    a file under ``<root path>/.gemseo-traces/<ClassName>/<n>.trace.yml``, ``<n>``
    being a 0-based counter local to ``ClassName``. The returned object id,
    e.g. ``"MDAJacobi/0"``, is both the registry-relative path to that entry
    and the value stamped onto the observee, so that another tracer created
    later for the same instance (e.g. the execution and linearization
    tracers of one discipline) reuses it instead of registering a second
    entry.

    This class has no notion of an execution root of its own: a caller aware
    of one, e.g. the directory manager, injects it with `set_root_path`. The
    root is not a constructor argument because `BaseMultiton` caches one
    instance per class, without arguments: the registry is built by whichever
    of the directory manager and a tracer comes first, so a root passed at
    construction would be dropped whenever a tracer wins that race.

    The per-class counters alone cannot make ids unique across processes:
    the lock below only serializes the threads of one process. A worker
    process created by forking (e.g. `multiprocessing` with the default
    start method on Linux) inherits a snapshot of the counters, so objects
    constructed in two forked workers would be assigned the same id. A
    worker created with a non-fork start method (the default on Windows and
    on macOS/Python 3.14+) rebuilds this registry from scratch, hence with
    counters starting over, while the root path still points to the root of
    the parent process (see `_rebuild_directory_manager`), so its entries
    would collide with those of the parent. Either way, the entries would
    overwrite each other under a shared ``.gemseo-traces`` directory, and the ids
    stamped onto two distinct observees would be the same.

    The index of an object registered in a worker process is therefore
    prefixed with the id of that process, e.g. ``"MDAJacobi/12345-0"``.
    Only an observee constructed in a worker is concerned: one unpickled
    from the parent carries the id stamped there, and the ids of the main
    process keep their bare index.
    """

    __traces_directory_name: ClassVar[str] = ".gemseo-traces"
    """The name of the directory holding the registry, under the root path.

    The registry lives in the same namespace as the execution directories the
    directory manager creates under the root path, and the latter are created
    with a bare `mkdir`, which raises when the directory already exists. The
    name therefore starts with a dot: an execution directory is named after
    the `secure_filename` of the observee, which strips the leading and
    trailing ``.`` and ``_`` from its output, so no observee name can produce
    this one.
    """

    __generation_counter: ClassVar[count[int]] = count()
    """The counter of registry instances constructed in the current process.

    Shared by every instance, unlike the per-class counter below: it survives
    the eviction of an instance from the `BaseMultiton` cache, so that the
    generation it hands out at construction time keeps increasing across
    resets instead of restarting from 0 along with the per-class counters.
    """

    __root_path: Path | None
    """The root directory under which registry entries are written.

    No entry is written while this is `None`, its default: only the
    per-class counter still advances, so that ids stay well-formed for
    tracers built directly, e.g. by unit tests that do not inject a root.
    """

    __class_name_to_count: dict[str, int]
    """The number of objects already registered, per sanitized class name."""

    __lock: Lock
    """The lock serializing the allocation of per-class indices."""

    __process_id: int
    """The id of the process that constructed this registry."""

    __generation: int
    """This instance's generation, i.e. its rank among the registries built in
    the current process so far.

    `BaseTracer` stamps it, alongside the object id, onto an observee it
    registers, so that it can later tell an id stamped by this very instance
    apart from one stamped by an instance since evicted by
    `Settings.__reset_directory_manager`, without keeping a reference to that
    (unpicklable, because of `__lock`) instance.
    """

    def __init__(self) -> None:  # noqa: D107
        self.__root_path = None
        self.__class_name_to_count = {}
        self.__lock = Lock()
        self.__process_id = getpid()
        self.__generation = next(self.__generation_counter)

    @property
    def generation(self) -> int:
        """The generation of this registry instance.

        Returns:
            This instance's rank among the registries built in the current
            process so far, see `__generation`.
        """
        return self.__generation

    def set_root_path(self, root_path: Path) -> None:
        """Set the root directory under which registry entries are written.

        Args:
            root_path: The root directory.
        """
        self.__root_path = root_path

    def register(
        self,
        class_name: str,
        *,
        name: str,
        documentation: str,
        init_arguments: Any,
    ) -> str:
        """Register an observed object and return its registry id.

        Args:
            class_name: The name of the observee's class.
            name: The name of the observee, empty if it has none.
            documentation: The observee's class documentation.
            init_arguments: The arguments used to instantiate the observee.

        Returns:
            The object id: the sanitized class name and the 0-based index of
            the observee among the objects of that class registered so far,
            e.g. ``"MDAJacobi/0"``, the index being prefixed with the id of
            the current process in a worker process, e.g.
            ``"MDAJacobi/12345-0"``.
        """
        # A class name made only of non-ASCII characters sanitizes to "":
        # fall back to the raw name rather than raising, since raising here
        # would abort the observee's construction instead of only degrading
        # its trace, at execution time.
        # Also, secure_filename strips leading underscores, so a private
        # class name (e.g. "_Foo") and its public counterpart ("Foo") share
        # one bucket and counter; harmless since no gemseo discipline, MDA,
        # scenario or algorithm class name is underscore-prefixed.
        sanitized_class_name = secure_filename(class_name) or class_name

        with self.__lock:
            index = self.__class_name_to_count.get(sanitized_class_name, 0)
            self.__class_name_to_count[sanitized_class_name] = index + 1

        entry_name = self.__get_entry_name(index)
        object_id = f"{sanitized_class_name}/{entry_name}"

        array_store = _NpyArrayStore(f"{entry_name}.trace.arrays")
        # Converted outside the block guarding the write below, so that a
        # conversion error, e.g. an unrepresentable constructor argument,
        # propagates and aborts the tracer construction: the caller then falls
        # back to a no-op tracer, see `BaseDMProcessor.__init__`. Converting here also
        # means the conversion happens, and can raise, even when
        # `__root_path` is `None`: only the write below is then skipped.
        converted_init_arguments = convert_to_yaml_data(init_arguments, array_store)

        if self.__root_path is not None:
            entry = _TraceRegistryEntry(
                object_id=object_id,
                type=class_name,
                name=name,
                documentation=documentation,
                init_arguments=converted_init_arguments,
            )
            try:
                self.__write(sanitized_class_name, entry_name, entry, array_store)
            except Exception:
                # An observee is registered while it is being constructed, so a
                # failure to write its entry, e.g. an unrepresentable value or a
                # full disk, shall degrade the trace instead of aborting that
                # construction. The id is returned nonetheless: the per-call
                # traces of the observee remain consistent with each other, they
                # merely point to a registry entry that does not exist.
                logger.exception(
                    "The trace registry entry of %s could not be written.", object_id
                )

        return object_id

    def __get_entry_name(self, index: int) -> str:
        """Return the name identifying an entry among those of its class.

        The name is the per-class index in the main process, and that index
        prefixed with the id of the current process in a worker, so that the
        entries of two processes sharing one ``.gemseo-traces`` directory cannot
        collide, see the class documentation.

        A worker is recognized either from `multiprocessing`, whichever start
        method created it, or from a change of process id since this registry
        was constructed, which also covers a bare `os.fork`, see
        [is_in_worker_process][gemseo.util._worker_context.is_in_worker_process].

        Args:
            index: The per-class index of the observee.

        Returns:
            The name of the entry.
        """
        if is_in_worker_process(self.__process_id):
            return f"{getpid()}-{index}"
        return str(index)

    def __write(
        self,
        sanitized_class_name: str,
        entry_name: str,
        entry: _TraceRegistryEntry,
        array_store: _NpyArrayStore,
    ) -> None:
        """Write a registry entry, and its referenced arrays, under the root path.

        The entry is written atomically, see `dump_yaml_to_file`, so that a
        reader of the registry, e.g. a tool consuming the traces while a run
        is ongoing, sees either no entry or a complete one; its arrays are
        written first, so that a reader that sees the entry also sees the
        arrays it references.

        Args:
            sanitized_class_name: The sanitized name of the observee's class.
            entry_name: The name identifying the entry among those of its class.
            entry: The registry entry to write.
            array_store: The store holding the entry's large constructor
                arguments, if any.
        """
        data = asdict(entry)
        if not data["name"]:
            # An observee without a name has no name entry at all, rather
            # than an empty one that asdict() would emit as "name: ''".
            del data["name"]
        directory_path: Path = (
            self.__root_path / self.__traces_directory_name / sanitized_class_name
        )
        # Instances of one class share this directory, unlike
        # DirectoryManager.start_directory's bare mkdir() for a single
        # execution directory.
        directory_path.mkdir(parents=True, exist_ok=True)
        array_store.write_arrays(directory_path)
        dump_yaml_to_file(data, directory_path / f"{entry_name}.trace.yml")
