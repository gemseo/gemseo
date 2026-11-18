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
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                         documentation
#        :author: Francois Gallard, Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""HDF5 file singleton used by the HDF5 cache."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from enum import StrEnum
from genericpath import exists
from importlib import metadata
from multiprocessing.util import Finalize
from os import getpid
from pathlib import Path
from time import perf_counter
from time import sleep
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Final
from weakref import WeakSet

import h5py
import numpy
from numpy import append
from numpy import array
from numpy import bytes_
from numpy import str_
from scipy.sparse import csr_array

from gemseo.core.cache.base import BaseCache
from gemseo.core.cache.util import hash_data
from gemseo.core.cache.util import to_real
from gemseo.util._compatibility.scipy import sparse_classes
from gemseo.util.multiprocessing.manager import get_multi_processing_manager
from gemseo.util.platform import platform_is_windows
from gemseo.util.singleton import SingleInstancePerFileAttribute

if TYPE_CHECKING:
    from collections.abc import Generator
    from multiprocessing.managers import DictProxy
    from multiprocessing.synchronize import RLock as RLockType

    from h5py import File

    from gemseo.util._compatibility.scipy import SparseArrayType
    from gemseo.util.typing import IntegerArray
    from gemseo.util.typing import StrKeyMapping
    from gemseo.util.typing import StrPath


logger = logging.getLogger(__name__)

_open_files: Final[set[HDF5FileSingleton]] = set()
"""The HDF files that are open."""

_file_singletons: Final[WeakSet[HDF5FileSingleton]] = WeakSet()
"""The HDF file singletons of the process.

This includes the copies of the singletons, e.g. unpickled or deep-copied,
which the registry of the singletons does not hold.
"""

_parent_files_kept_alive: Final[list[File]] = []
"""The HDF file handles inherited from the parent process, never closed.

The parent process closes its files before forking,
except the ones that an operation uses, e.g. in another thread.
A forked process shall not close them,
otherwise HDF5 could write stale metadata in the files of the parent process;
they are referenced here so that the garbage collector does not close them either.
"""

_processes_closing_files: Final[set[int]] = set()
"""The identifiers of the processes that close their open HDF files at exit."""


def _close_files() -> None:
    """Close the HDF files still open."""
    # Closing a file removes it from _open_files,
    # which shall not change size while being iterated over.
    for file in list(_open_files):
        try:
            file.close()
        except Exception:  # noqa: BLE001, PERF203 # pragma: no cover
            # Closing a file shall not prevent closing the other ones,
            # and an exception raised at interpreter shutdown is of no use.
            logger.debug("Failed to close the HDF file %s.", file.hdf_file_path)


def _close_files_at_process_exit() -> None:
    """Close the HDF files still open at the end of the current process.

    The files are closed by a finalizer of `multiprocessing`,
    which runs at the end of the process,
    including when an exception is raised by the process:
    from `atexit` in a main or spawned process,
    and when a worker process forked by `multiprocessing` exits with `os._exit`,
    which does not run `atexit`.
    A forked process clears the finalizers inherited from its parent though,
    so a finalizer is registered once in each process opening a file.
    """
    process_id = getpid()
    if process_id not in _processes_closing_files:
        _processes_closing_files.add(process_id)
        Finalize(None, _close_files, exitpriority=0)


def _forget_parent_process_state() -> None:
    """Make the HDF file singletons forget the state of the parent process.

    A forked process inherits the singletons of its parent
    without going through their pickling.
    """
    # This loop does not change _file_singletons,
    # and a WeakSet defers the removals made by the garbage collector
    # until the end of its iteration;
    # the copy only keeps the singletons alive until the end of the loop.
    for singleton in list(_file_singletons):
        singleton._forget_parent_process_state()


def _close_idle_files() -> None:
    """Close the HDF files that no operation uses, before forking the process.

    A forked process would inherit the open files and their HDF5 locks,
    which would prevent the worker processes from opening them;
    the parent process opens them again at its next operation.
    """
    # Closing a file removes it from _open_files,
    # which shall not change size while being iterated over.
    for singleton in list(_open_files):
        singleton._close_if_idle()


if not platform_is_windows:  # pragma: no cover
    # Windows cannot fork a process.
    # mypy does not narrow the platform from platform_is_windows,
    # so it does not know that os.register_at_fork exists here
    # when it checks the code on Windows.
    os.register_at_fork(  # type: ignore[attr-defined, unused-ignore]
        before=_close_idle_files, after_in_child=_forget_parent_process_state
    )


class HDF5FileSingleton(metaclass=SingleInstancePerFileAttribute):
    """Singleton to access an HDF file.

    Used for multithreading/multiprocessing access with a lock.
    """

    # We create a single instance of cache per HDF5 file
    # to allow the multiprocessing lock to work
    # Ie we want a single lock even if we instantiate multiple
    # caches

    hdf_file_path: str
    """The path to the HDF file."""

    _lock: RLockType
    """The lock used for multiprocessing."""

    hash_tag: ClassVar[str] = "hash"
    """The label for the hash."""

    last_index_tag: ClassVar[str] = "last_index"
    """The label of the attribute of a node storing the last index allocated."""

    file_format_version: ClassVar[int] = 3
    """The version of the file format.

    The version 3, introduced by GEMSEO 7,
    marks the files whose hashes have been computed by
    [hash_data][gemseo.core.cache.util.hash_data]
    from the canonical form of the input data;
    the files of an older version,
    written by GEMSEO 6 or earlier,
    store hashes computed from the raw input data,
    which are computed again when such a file is opened.
    """

    __retry_period: ClassVar[float] = 0.01
    """The number of seconds between two attempts to open a locked file."""

    _file_kept_open: bool
    """Whether to keep the file open between two operations.

    Set it through `HDF5Cache.file_kept_open`.
    It is reset to `False` in a worker process,
    which unpickles or inherits the singleton.
    """

    _timeout: float
    """The maximum number of seconds to wait for a file locked by another process.

    Set it through `HDF5Cache.timeout`.
    """

    __n_keep_open: int
    """The number of nested `keep_open()` contexts keeping the file open."""

    __file: File | None
    """The hdf5 file handle."""

    __compute_hashes: bool
    """Whether to compute the hashes of the entries instead of reading them.

    This is the case of a file of an older version that cannot be updated,
    whose stored hashes may have been computed by another hash function.
    """

    __n_operations: int
    """The number of operations using the file, which shall not close it."""

    class __SparseMatricesAttribute(StrEnum):  # noqa: N801
        """Attribute name required to store sparse matrices in CSR format."""

        SPARSE = "sparse"
        INDICES = "indices"
        INDPTR = "indptr"
        SHAPE = "shape"

    def __init__(
        self,
        hdf_file_path: str,
    ) -> None:
        """
        Args:
            hdf_file_path: The path to the HDF file.
        """  # noqa: D205, D212, D415
        self.__n_keep_open = 0
        self._file_kept_open = False
        self._timeout = 30.0
        self.__file = None
        self.__compute_hashes = False
        self.__n_operations = 0
        self.hdf_file_path = hdf_file_path
        _file_singletons.add(self)
        if self.__get_file_format_version() < self.file_format_version:
            self.__update_file_format()

        self._lock = get_multi_processing_manager().RLock()

    @property
    def lock(self) -> RLockType:
        """The lock used for multiprocessing."""
        return self._lock

    def write_data(
        self,
        data: StrKeyMapping,
        group: BaseCache.Group,
        index: int,
        hdf_node_path: str,
        data_hash: int | None = None,
    ) -> None:
        """Cache input data to avoid re-evaluation.

        Args:
            data: The data containing the values of the names to cache.
            group: The group.
            index: The index of the entry in the cache.
            hdf_node_path: The name of the HDF group to store the entries,
                possibly passed as a path `root_name/.../group_name/.../node_name`.
            data_hash: The hash of the input data;
                if `None`,
                computed from `data` when the entry has no hash yet.

        Raises:
            TypeError: If a value is an array of Python objects,
                e.g. `None`.
            RuntimeError: If the data cannot be written in the file.
        """
        self.__write_data(data, group, index, hdf_node_path, data_hash)

    def write_new_entry(
        self,
        data: StrKeyMapping,
        hdf_node_path: str,
        data_hash: int | None = None,
    ) -> int:
        """Write the input data of a new entry, at an index allocated from the file.

        The caches of the different processes
        cannot allocate a unique index on their own,
        so the write creating an entry lets the file do it.

        Args:
            data: The input data.
            hdf_node_path: The name of the HDF group to store the entries,
                possibly passed as a path `root_name/.../group_name/.../node_name`.
            data_hash: The hash of the input data;
                if `None`,
                computed from `data`.

        Returns:
            The index of the entry.

        Raises:
            TypeError: If a value is an array of Python objects,
                e.g. `None`.
            RuntimeError: If the data cannot be written in the file.
        """
        return self.__write_data(
            data, BaseCache.Group.INPUTS, None, hdf_node_path, data_hash
        )

    def __write_data(
        self,
        data: StrKeyMapping,
        group: BaseCache.Group,
        index: int | None,
        hdf_node_path: str,
        data_hash: int | None,
    ) -> int:
        """Write data in the file.

        Args:
            data: The data containing the values of the names to cache.
            group: The group.
            index: The index of the entry in the cache;
                if `None`, allocate a new one from the file.
            hdf_node_path: The name of the HDF group to store the entries,
                possibly passed as a path `root_name/.../group_name/.../node_name`.
            data_hash: The hash of the input data;
                if `None`,
                computed from `data` when the entry has no hash yet.

        Returns:
            The index of the entry.

        Raises:
            TypeError: If a value is an array of Python objects,
                e.g. `None`.
            RuntimeError: If the data cannot be written in the file.
        """
        # Convert and validate all the values before touching the file,
        # so that a failure leaves no partial entry.
        converted_data = self.__convert_data(data)

        with self.__open(mode="a"):
            assert self.__file is not None

            if not len(self.__file):
                self.__set_file_format_version(self.__file)

            root = self.__file.require_group(hdf_node_path)
            is_allocated = index is None
            if index is None:
                index = self.__allocate_index(root)

            entry = root.get(str(index))
            is_new_entry = entry is None
            if (entry is None or entry.get(self.hash_tag) is None) and (
                data_hash is None
            ):
                data_hash = hash_data(data)

            entry = root.require_group(str(index))
            entry_group = entry.require_group(group)

            try:
                # Write hash if needed
                if entry.get(self.hash_tag) is None:
                    entry.create_dataset(
                        self.hash_tag, data=array([data_hash], dtype="bytes")
                    )

                for name, value in converted_data.items():
                    if isinstance(value, sparse_classes):
                        self.__write_sparse_array(entry_group, name, value)
                    else:
                        entry_group[name] = value

            except (RuntimeError, OSError, ValueError):
                if is_new_entry:
                    # A partial entry would be read as a real one.
                    del root[str(index)]

                msg = "Failed to cache dataset %s.%s.%s in file: %s"
                raise RuntimeError(
                    msg,
                    hdf_node_path,
                    index,
                    entry_group,
                    self.hdf_file_path,
                ) from None

            if is_allocated:
                # The index is recorded once the entry is written,
                # so that a failed write does not leave a gap.
                root.attrs[self.last_index_tag] = index

        return index

    def __allocate_index(self, node: h5py.Group) -> int:
        """Allocate the index of a new entry of a node.

        The last index allocated is stored in an attribute of the node,
        once the entry is written,
        so that the names of all its entries are not read at each allocation.

        Args:
            node: The node.

        Returns:
            The index of the new entry.
        """
        last_index = node.attrs.get(self.last_index_tag)
        if last_index is None:
            # The node has been written without this attribute,
            # so its entries are read once.
            # It can also hold the nodes of other caches,
            # which are not named after an index.
            last_index = max((int(name) for name in node if name.isdigit()), default=0)

        index = int(last_index) + 1
        # An entry can have been written at a given index without allocation.
        while str(index) in node:
            index += 1

        return index

    @staticmethod
    def __convert_data(data: StrKeyMapping) -> dict[str, Any]:
        """Convert the values to what can be stored in an HDF5 file.

        Args:
            data: The data containing the values of the names to cache.

        Returns:
            The converted values.

        Raises:
            TypeError: If a value is an array of Python objects.
        """
        converted_data: dict[str, Any] = {}
        for name, value in data.items():
            if isinstance(value, sparse_classes):
                converted_data[name] = value
            elif value.dtype.kind == "O":
                if value.size == 1 and value.item() is None:
                    msg = (
                        f"The value of {name} is None, "
                        "which cannot be cached in an HDF5 file."
                    )
                else:
                    msg = (
                        f"The value of {name} is an array of Python objects, "
                        "which cannot be cached in an HDF5 file."
                    )
                raise TypeError(msg)
            elif value.dtype.type is str_:
                # See https://docs.h5py.org/en/stable/strings.html
                converted_data[name] = numpy.char.encode(value, "utf-8")
            else:
                converted_data[name] = to_real(value)
        return converted_data

    def __write_sparse_array(
        self,
        group: h5py.Group,
        dataset_name: str,
        value: SparseArrayType,
    ) -> None:
        """Store sparse array in HDF5 group.

        Args:
            group: The group.
            dataset_name: The name of the dataset to store the sparse array in.
            value: The sparse array.
        """
        value = value.tocsr()

        # Create a dataset containing the non-zero elements of value
        dataset = group.create_dataset(dataset_name, data=to_real(value.data))

        # Add as attribute the indices, pointers and shape required for reconstruction
        dataset.attrs.create(self.__SparseMatricesAttribute.INDICES, value.indices)
        dataset.attrs.create(self.__SparseMatricesAttribute.INDPTR, value.indptr)
        dataset.attrs.create(self.__SparseMatricesAttribute.SHAPE, value.shape)

        # Add a sparse flag
        dataset.attrs.create(self.__SparseMatricesAttribute.SPARSE, True)

    def read_data(
        self,
        index: int,
        group: BaseCache.Group,
        hdf_node_path: str,
    ) -> StrKeyMapping:
        """Read the data for given index and group.

        Args:
            index:  The index of the entry.
            group: The group.
            hdf_node_path: The name of the HDF group where the entries are stored,
                possibly passed as a path `root_name/.../group_name/.../node_name`.

        Returns:
            The group data and the input data hash.

        Raises:
            ValueError: If the group cannot be found.
        """
        with self.__open():
            assert self.__file is not None
            root = self.__file[hdf_node_path]

            if not self._has_group(index, group, hdf_node_path):
                return {}

            entry = root[str(index)]

            data = {}
            for key, dataset in entry[group].items():
                if dataset.attrs.get(self.__SparseMatricesAttribute.SPARSE):
                    value = self.__read_sparse_array(dataset)
                elif dataset.dtype.type == bytes_:
                    value = numpy.char.decode(array(dataset), "utf-8")
                else:
                    value = array(dataset)
                data[key] = value

        return data

    def __read_sparse_array(self, dataset: h5py.Dataset) -> csr_array:
        """Read sparse array from a HDF5 dataset.

        Args:
            dataset: The dataset to the read the data from.

        Returns:
            The sparse array in CSR format.
        """
        indices = dataset.attrs.get(self.__SparseMatricesAttribute.INDICES)
        indptr = dataset.attrs.get(self.__SparseMatricesAttribute.INDPTR)
        shape = dataset.attrs.get(self.__SparseMatricesAttribute.SHAPE)

        return csr_array((dataset, indices, indptr), shape)

    def _has_group(
        self,
        index: int,
        group: BaseCache.Group,
        hdf_node_path: str,
    ) -> bool:
        """Return whether a group exists.

        Args:
            index: The index of the entry.
            group: The group.
            hdf_node_path: The name of the HDF group where the entries are stored,
                possibly passed as a path `root_name/.../group_name/.../node_name`.

        Returns:
            Whether a group exists.
        """
        assert self.__file is not None
        entry = self.__file[hdf_node_path].get(str(index))
        if entry is None:
            return False
        return entry.get(group) is not None

    def has_group(
        self,
        index: int,
        group: BaseCache.Group,
        hdf_node_path: str,
    ) -> bool:
        """Check if an entry has data corresponding to a given group.

        Args:
            index: The index of the entry.
            group: The group.
            hdf_node_path: The name of the HDF group where the entries are stored,
                possibly passed as a path `root_name/.../group_name/.../node_name`.

        Returns:
            Whether the entry has data for this group.
        """
        with self.__open():
            return self._has_group(index, group, hdf_node_path)

    def read_hashes(
        self,
        hash_to_indices: DictProxy[int, IntegerArray],
        hdf_node_path: str,
    ) -> int:
        """Read the hashes in the HDF file.

        Args:
            hash_to_indices: The indices associated to the hashes.
            hdf_node_path: The name of the HDF group where the entries are stored,
                possibly passed as a path `root_name/.../group_name/.../node_name`.

        Returns:
            The maximum index.
        """
        if not exists(self.hdf_file_path):
            return 0

        # We must lock so that no data is added to the cache meanwhile
        with self.__open():
            assert self.__file is not None
            root = self.__file.get(hdf_node_path)

            if root is None:
                return 0

            max_index = 0

            for index, entry in root.items():
                index = int(index)
                if self.__compute_hashes:
                    hash_ = self.__compute_hash(entry)
                else:
                    hash_ = int(array(entry[self.hash_tag])[0])
                indices = hash_to_indices.get(hash_)

                if indices is None:
                    hash_to_indices[hash_] = array([index])
                elif index not in indices:
                    # The map can be shared with a cache that has read it already,
                    # e.g. by the copy of a cache unpickled by a worker process.
                    hash_to_indices[hash_] = append(indices, array([index]))

                max_index = max(max_index, index)

        return max_index

    def clear(
        self,
        hdf_node_path: str,
    ) -> None:
        """Clear a node.

        Args:
            hdf_node_path: The name of the HDF group to clear,
                possibly passed as a path `root_name/.../group_name/.../node_name`.
        """
        with self.__open(mode="a"):
            assert self.__file is not None
            del self.__file[hdf_node_path]

    def __get_file_format_version(self) -> int:
        """Return the version of the file format, checking that it can be handled.

        Returns:
            The version of the file format,
            which is the current one for a file that does not exist or is empty.

        Raises:
            ValueError: If the version of the file format is missing or
                greater than the current one.
        """
        if not Path(self.hdf_file_path).exists():
            return self.file_format_version

        with self.__open():
            assert self.__file is not None
            if not len(self.__file):
                return self.file_format_version
            version = self.__file.attrs.get("version")

        if version is None:
            msg = (
                f"The file {self.hdf_file_path} cannot be used because it has no file "
                "format version: see HDF5Cache.update_file_format to convert it."
            )
            raise ValueError(msg)

        if version > self.file_format_version:
            # update_file_format cannot convert a file to an older version.
            msg = (
                f"The file {self.hdf_file_path} cannot be used because its file "
                f"format version is {version} "
                f"while the expected version is {self.file_format_version}: "
                f"a version of GEMSEO newer than {metadata.version('gemseo')} "
                "is needed to use it."
            )
            raise ValueError(msg)

        return int(version)

    def __update_file_format(self) -> None:
        """Update the format of a file of an older version.

        The hashes stored by an older version have been computed
        with another hash function, so they would never be found again.
        A file that cannot be written, e.g. a read-only reference cache,
        is left unchanged and the hashes of its entries are computed when read.

        Warning:
            The update is not safe when several processes open the same file
            at the same time: nothing serializes them across processes.
        """
        logger.info(
            "Updating the format of the file %s to the version %s.",
            self.hdf_file_path,
            self.file_format_version,
        )
        try:
            self.update_file_format(self.hdf_file_path)
        except OSError:
            logger.warning(
                "The format of the file %s cannot be updated, "
                "so the hashes of all its entries are computed "
                "whenever it is opened, "
                "including by each process of a parallel execution; "
                "make it writable and open it once to avoid this cost.",
                self.hdf_file_path,
            )
            self.__compute_hashes = True

    @classmethod
    def __set_file_format_version(
        cls,
        h5_file: h5py.File,
    ) -> None:
        """Change the version of an HDF5 file to `file_format_version`.

        Args:
            h5_file: A HDF5 file object.
        """
        h5_file.attrs["version"] = cls.file_format_version

    @classmethod
    def update_file_format(
        cls,
        hdf_file_path: StrPath,
    ) -> None:
        """Update the format of a HDF5 file.

        GEMSEO 3.2.0 added a `HDF5FileSingleton.file_format_version`
        to the HDF5 files,
        to allow handling its maintenance and evolutions.
        In particular,
        GEMSEO 3.2.0 fixed the hashing of the data dictionaries.

        The hashes of the entries are computed again,
        as the hashing of the data has changed since,
        and the file is marked with the current version of the file format.

        Args:
            hdf_file_path: A HDF5 file path.
        """
        with h5py.File(str(hdf_file_path), "a") as h5file:
            # A node can be passed as a path `root_name/.../node_name`,
            # so the entries are looked for at any depth;
            # they are collected first, as the file is not modified while visited.
            entry_names = []
            h5file.visititems(
                lambda name, obj: (
                    entry_names.append(name) if cls.__is_entry(name, obj) else None
                )
            )
            for entry_name in entry_names:
                entry = h5file[entry_name]
                data_hash = array([cls.__compute_hash(entry)], dtype="bytes")
                # The dataset has the width of the old hash,
                # which cuts a longer new hash if assigned into it.
                # It can be missing after an update interrupted at this point.
                if cls.hash_tag in entry:
                    del entry[cls.hash_tag]

                entry.create_dataset(cls.hash_tag, data=data_hash)

            # The version is set last,
            # so that a failure leaves the file at its old version.
            cls.__set_file_format_version(h5file)

    @staticmethod
    def __compute_hash(entry: h5py.Group) -> int:
        """Compute the hash of an entry from its input data.

        Args:
            entry: The entry.

        Returns:
            The hash of the entry.
        """
        data = entry[BaseCache.Group.INPUTS]
        return hash_data({key: array(val) for key, val in data.items()})

    @classmethod
    def __is_entry(cls, name: str, obj: h5py.Group | h5py.Dataset) -> bool:
        """Return whether an object of an HDF file is an entry of a cache.

        An entry is a group named after its index and holding input data.
        Its hash is not required,
        so that an update interrupted between its deletion and its creation
        can be completed.

        Args:
            name: The path of the object in the file.
            obj: The object.

        Returns:
            Whether the object is an entry of a cache.
        """
        return (
            isinstance(obj, h5py.Group)
            and name.rpartition("/")[2].isdigit()
            and BaseCache.Group.INPUTS in obj
        )

    @contextmanager
    def __open(self, mode: str = "r") -> Generator[None, None, None]:
        """Open a hdf5 file.

        The file handle is used via `__file`.

        Args:
            mode: The opening mode, see `h5py.File`.

        Raises:
            RuntimeError: When the file shall be opened again in append mode
                while an operation reads it,
                as this would invalidate the handles of this operation.
        """
        # A file opened in append mode can also be read.
        if self.__file is None or (mode == "a" and self.__file.mode != "r+"):
            if self.__n_operations:
                msg = (
                    f"The HDF file {self.hdf_file_path} cannot be written "
                    "while it is read."
                )
                raise RuntimeError(msg)

            self.close()
            self.__file = self.__create_file(mode)
            _open_files.add(self)
            _close_files_at_process_exit()

        # The file shall be closed whatever happens in the body of the caller,
        # including when it returns or raises an exception,
        # otherwise HDF5 keeps a lock preventing another process
        # from opening the file,
        # but only by the last operation using it,
        # e.g. when another thread or an enclosing operation still uses it.
        self.__n_operations += 1
        try:
            yield
        finally:
            self.__n_operations -= 1
            if not self.__n_operations and not self.__keeps_open:
                self.close()

    @property
    def __keeps_open(self) -> bool:
        """Whether the file is kept open when leaving `__open()`."""
        return bool(self.__n_keep_open) or self._file_kept_open

    def __create_file(self, mode: str) -> File:
        """Open the HDF file, waiting for the other processes to release it.

        Unlike a database engine, HDF5 has no way of waiting for a lock:
        a process opening a file that another one is using fails immediately.
        The opening is therefore retried until `_timeout` is exceeded,
        so that the processes writing in the same file
        do not have to open it at different times.

        Args:
            mode: The opening mode, see `h5py.File`.

        Returns:
            The HDF file object.

        Raises:
            OSError: If the file is still locked by another process
                after `_timeout` seconds.
        """
        deadline = perf_counter() + self._timeout
        while True:
            try:
                return h5py.File(self.hdf_file_path, mode=mode)
            except OSError as error:  # noqa: PERF203
                if "unable to lock file" not in str(error):
                    raise

                if perf_counter() >= deadline:
                    logger.warning(
                        "The HDF file %s is still locked after %s seconds.",
                        self.hdf_file_path,
                        self._timeout,
                    )
                    raise

                sleep(self.__retry_period)

    @contextmanager
    def keep_open(self) -> Generator[None, None, None]:
        """Keep the file open for all file operations done in this context manager."""
        self.__n_keep_open += 1
        try:
            yield
        finally:
            self.__n_keep_open -= 1
            # An enclosing context keeps the file open
            # and an operation still using the file closes it when it ends.
            if not self.__keeps_open and not self.__n_operations:
                self.close()

    def _forget_parent_process_state(self) -> None:
        """Forget the state of the parent process, in a worker process.

        Keeping the file open is a choice of the parent process
        that would make the worker processes lock the file out of each other.
        """
        self.__n_keep_open = 0
        self._file_kept_open = False
        # The operations of the parent process do not run in this one.
        self.__n_operations = 0
        if self.__file is not None:
            # The handle of the parent process is dropped without being closed.
            _parent_files_kept_alive.append(self.__file)
            self.__file = None
            _open_files.discard(self)

    def _close_if_idle(self) -> None:
        """Close the file handle if no operation uses it."""
        if not self.__n_operations:
            self.close()

    def close(self) -> None:
        """Close the file handle.

        Do nothing if the file is already closed.
        The file is opened again by the next read or write.

        Warning:
            This closes the file even when an operation uses it,
            e.g. in another thread.
        """
        try:
            if self.__file is not None:
                self.__file.close()
        finally:
            # A handle that fails to close is forgotten all the same,
            # so that the file is opened again by the next read or write.
            self.__file = None
            _open_files.discard(self)

    def __getstate__(self):
        # Another process is about to use the file, e.g. because the cache is sent to
        # a sub-process, and HDF5 would prevent it from opening a file left open here;
        # a file used by an operation, e.g. of another thread, is closed by it.
        self._close_if_idle()
        state = self.__dict__.copy()
        # The hd5py file handle is not pickable.
        del state["_HDF5FileSingleton__file"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # The pickled state has no file handle.
        self.__file = None
        _file_singletons.add(self)
        # The singleton is unpickled by another process, e.g. a worker process.
        self._forget_parent_process_state()
