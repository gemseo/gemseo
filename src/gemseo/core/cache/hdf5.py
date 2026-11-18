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
"""Caching module to store all the entries in an HDF file."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Literal
from typing import overload

from gemseo.core.cache._hdf5_file_singleton import HDF5FileSingleton
from gemseo.core.cache.base_full import BaseFullCache
from gemseo.core.cache.util import hash_data
from gemseo.util.data_conversion import nest_flat_bilevel_dict
from gemseo.util.lock import synchronized

if TYPE_CHECKING:
    from gemseo.core.data_converter.base import BaseDataConverter
    from gemseo.util.string import MultiLineString
    from gemseo.util.typing import JacobianData
    from gemseo.util.typing import MutableStrKeyMapping
    from gemseo.util.typing import StrKeyMapping
    from gemseo.util.typing import StrPath

logger = logging.getLogger(__name__)


class HDF5Cache(BaseFullCache):
    """Cache using disk HDF5 file to store the data.

    The data, either input or output, to be stored in the cache shall be
    castable to a NumPy array. Otherwise, data converters shall be provided when
    creating an instance.

    Several processes can write to the same node of the same file,
    e.g. when a DOE evaluates a discipline with several processes:
    the accesses to the file are serialized by the lock of the cache
    and the index of a new entry is allocated from the file,
    and not by the cache of a given process.

    The file is opened and closed at every read and write,
    which costs a lot of time.
    When a single process uses the file,
    set [file_kept_open][gemseo.core.cache.hdf5.HDF5Cache.file_kept_open]
    to keep it open between two operations:

    ```python
    cache.file_kept_open = True
    ```

    The file is then closed at the end of the process,
    whether or not it raises an exception,
    and can be closed earlier with
    [close_file()][gemseo.core.cache.hdf5.HDF5Cache.close_file].

    HDF5 forbids a process to open a file that another one has open,
    so opening a file used by another process,
    e.g. a worker process or an HDF viewer,
    waits for its release
    up to [timeout][gemseo.core.cache.hdf5.HDF5Cache.timeout] seconds,
    30 by default,
    then raises an `OSError`.
    """

    __input_data_converter: BaseDataConverter | None
    """The data converter used to convert input data."""

    __output_data_converter: BaseDataConverter | None
    """The data converter used to convert output data."""

    def __init__(
        self,
        tolerance: float = 0.0,
        name: str = "",
        hdf_file_path: StrPath = "cache.hdf5",
        hdf_node_path: str = "node",
        input_data_converter: BaseDataConverter | None = None,
        output_data_converter: BaseDataConverter | None = None,
    ) -> None:
        """
        Args:
            name: A name for the cache.
                If empty, use `hdf_node_path`.
            hdf_file_path: The path of the HDF file.
                Initialize a singleton to access the HDF file.
                This singleton is used for multithreading/multiprocessing access
                with a lock.
            hdf_node_path: The name to the HDF node,
                possibly passed as a path `root_name/.../group_name/.../node_name`.
            input_data_converter: The data converter to convert the input data.
            output_data_converter: The data converter to convert the output data.

        Warning:
            This class relies on some multiprocessing features, it is therefore
            necessary to protect its execution with an `if __name__ == '__main__':`
            statement when working on Windows.

            A file written by GEMSEO 6 or earlier
            is rewritten in place when it is opened,
            with the hashes of its entries computed again;
            GEMSEO 6 and earlier can no longer open it.
            This rewrite is not safe when several processes open the file
            at the same time,
            so such a file shall be opened once by a single process first.
            A file that cannot be written is left unchanged
            and the hashes of its entries are computed whenever it is opened.
        """  # noqa: D205, D212, D415
        super().__init__(tolerance, name or hdf_node_path)
        self.__input_data_converter = input_data_converter
        self.__output_data_converter = output_data_converter
        self.__hdf_node_path = hdf_node_path
        self.__hdf_file = HDF5FileSingleton(str(hdf_file_path))
        self._lock = self.__hdf_file.lock
        self._read_hashes()

    @property
    def hdf_file(self) -> HDF5FileSingleton:
        """The HDF file handler."""
        return self.__hdf_file

    @property
    def hdf_node_path(self) -> str:
        """The path to the HDF node."""
        return self.__hdf_node_path

    @property
    def file_kept_open(self) -> bool:
        """Whether to keep the HDF file open between two operations.

        Opening and closing the file at every read and write costs a lot of time,
        which this option avoids when a single process uses the file.
        The data are written to the disk when the file is closed,
        at the latest at the end of the process,
        so a process killed without exiting, e.g. with `SIGKILL`,
        can leave the file corrupted.

        This option applies to all the caches sharing the HDF file.

        Warning:
            HDF5 forbids a process to open a file that another one has open,
            so this option is reset to `False` in a worker process,
            e.g. when a DOE evaluates a discipline with several processes.
        """
        return self.__hdf_file._file_kept_open

    @file_kept_open.setter
    def file_kept_open(self, value: bool) -> None:
        self.__hdf_file._file_kept_open = value

    @property
    def timeout(self) -> float:
        """The maximum number of seconds to wait for a file locked by another process.

        HDF5 forbids a process to open a file that another one has open,
        e.g. a worker process or an HDF viewer,
        so opening it is retried until it is released or this time has elapsed;
        an `OSError` is raised in the latter case.

        This option applies to all the caches sharing the HDF file,
        including in the worker processes.
        """
        return self.__hdf_file._timeout

    @timeout.setter
    def timeout(self, value: float) -> None:
        self.__hdf_file._timeout = value

    def close_file(self) -> None:
        """Close the HDF file.

        Do nothing if the file is already closed.
        The file is opened again by the next read or write.
        """
        self.__hdf_file.close()

    def _get_string_representation(self) -> MultiLineString:
        mls = super()._get_string_representation()
        mls.add("HDF file path: {}", self.__hdf_file.hdf_file_path)
        mls.add("HDF node path: {}", self.__hdf_node_path)
        return mls

    def _copy_empty_cache(self) -> HDF5Cache:
        file_path = Path(self.__hdf_file.hdf_file_path)
        return self.__class__(
            hdf_file_path=file_path.parent / ("new_" + file_path.name),
            hdf_node_path=self.__hdf_node_path,
            tolerance=self._tolerance,
            name=self.name,
            input_data_converter=self.__input_data_converter,
            output_data_converter=self.__output_data_converter,
        )

    @synchronized
    def _read_hashes(self) -> None:
        """Read the hashes dict in the HDF file."""
        max_index = self.__hdf_file.read_hashes(
            self._hash_to_indices, self.__hdf_node_path
        )
        self._last_accessed_index.value = max_index
        self._max_index.value = max_index
        cache_size = len(self._hash_to_indices)
        if cache_size > 0:
            msg = "Found %s entries in the cache file : %s node : %s"
            logger.info(
                msg, cache_size, self.__hdf_file.hdf_file_path, self.__hdf_node_path
            )

    def _has_group(
        self,
        index: int,
        group: BaseFullCache.Group,
    ) -> bool:
        return self.__hdf_file.has_group(index, group, self.__hdf_node_path)

    @synchronized
    def clear(self) -> None:  # noqa:D102
        super().clear()
        self.__hdf_file.clear(self.__hdf_node_path)

    @overload
    def _read_data(
        self,
        index: int,
        group: Literal[BaseFullCache.Group.INPUTS, BaseFullCache.Group.OUTPUTS],
    ) -> StrKeyMapping: ...

    @overload
    def _read_data(
        self,
        index: int,
        group: Literal[BaseFullCache.Group.JACOBIAN],
    ) -> JacobianData: ...

    def _read_data(
        self,
        index: int,
        group: BaseFullCache.Group,
    ) -> StrKeyMapping | JacobianData:
        data = self.__hdf_file.read_data(index, group, self.__hdf_node_path)
        if not data:
            # Fast path to avoid converting empty data in the next conditional blocks.
            return data
        if self.__input_data_converter is not None and group == self.Group.INPUTS:
            to_value = self.__input_data_converter.convert_array_to_value
            for input_name, value in data.items():
                data[input_name] = to_value(input_name, value)
        elif self.__output_data_converter is not None and group == self.Group.OUTPUTS:
            to_value = self.__output_data_converter.convert_array_to_value
            for output_name, value in data.items():
                data[output_name] = to_value(output_name, value)
        elif group == self.Group.JACOBIAN:
            data = nest_flat_bilevel_dict(data, separator=self._jacobian_separator)
        return data

    def _hash_input_data(
        self,
        input_data: StrKeyMapping,
    ) -> int:
        # The file stores the hashes of the input data converted to arrays,
        # which its update and the read-only fallback compute again
        # from the input data stored in the file,
        # so the lookups hash the input data in the same form.
        if self.__input_data_converter is not None:
            to_array = self.__input_data_converter.convert_value_to_array
            # A None value is left as is, as hash_data ignores it
            # and the file refuses it with a clear error when writing.
            input_data = {
                name: value if value is None else to_array(name, value)
                for name, value in input_data.items()
            }

        return hash_data(input_data)

    def _write_data(
        self,
        values: MutableStrKeyMapping,
        group: BaseFullCache.Group,
        index: int,
    ) -> None:
        # The input data are written by _write_input_data, with their hash.
        self.__hdf_file.write_data(
            self.__convert_to_arrays(values, group), group, index, self.__hdf_node_path
        )

    def _write_input_data(
        self,
        input_data: StrKeyMapping,
        index: int,
        data_hash: int,
    ) -> int:
        # A cache counts the entries in a counter of its own process,
        # so two processes writing in the same node would pick the same index;
        # the file allocates it instead, whatever the index counted by the cache.
        return self.__hdf_file.write_new_entry(
            self.__convert_to_arrays(input_data, self.Group.INPUTS),
            self.__hdf_node_path,
            data_hash,
        )

    def __convert_to_arrays(
        self,
        values: StrKeyMapping,
        group: BaseFullCache.Group,
    ) -> StrKeyMapping:
        """Convert the data of a group to arrays, with the data converter if any.

        Args:
            values: The data.
            group: The group.

        Returns:
            The data converted to arrays.
        """
        if group == self.Group.INPUTS:
            converter = self.__input_data_converter
        elif group == self.Group.OUTPUTS:
            converter = self.__output_data_converter
        else:
            converter = None
        if converter is None:
            return values

        to_array = converter.convert_value_to_array
        return {name: to_array(name, value) for name, value in values.items()}

    @staticmethod
    def update_file_format(
        hdf_file_path: StrPath,
    ) -> None:
        """Update the format of a HDF5 file.

        See Also:
            `gemseo.core.cache._hdf5_file_singleton.HDF5FileSingleton.update_file_format()`.

        Args:
            hdf_file_path: A HDF5 file path.
        """
        HDF5FileSingleton.update_file_format(hdf_file_path)

    def __setstate__(self, state):
        super().__setstate__(state)
        self._read_hashes()
