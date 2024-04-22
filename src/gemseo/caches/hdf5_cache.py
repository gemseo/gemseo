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

from gemseo.caches._hdf5_file_singleton import HDF5FileSingleton
from gemseo.caches.base_full_cache import BaseFullCache
from gemseo.caches.cache_entry import CacheEntry
from gemseo.utils.data_conversion import nest_flat_bilevel_dict
from gemseo.utils.locks import synchronized

if TYPE_CHECKING:
    from collections.abc import Iterator
    from multiprocessing.synchronize import RLock as RLockType

    from gemseo.typing import JacobianData
    from gemseo.typing import StrKeyMapping
    from gemseo.utils.string_tools import MultiLineString

LOGGER = logging.getLogger(__name__)


class HDF5Cache(BaseFullCache):
    """Cache using disk HDF5 file to store the data."""

    def __init__(
        self,
        tolerance: float = 0.0,
        name: str | None = None,
        hdf_file_path: str | Path = "cache.hdf5",
        hdf_node_path: str = "node",
    ) -> None:
        """
        Args:
            name: A name for the cache.
                If ``None``, use :attr:`hdf_node_path``.
            hdf_file_path: The path of the HDF file.
                Initialize a singleton to access the HDF file.
                This singleton is used for multithreading/multiprocessing access
                with a lock.
            hdf_node_path: The name to the HDF node,
                possibly passed as a path ``root_name/.../group_name/.../node_name``.

        Warnings:
            This class relies on some multiprocessing features, it is therefore
            necessary to protect its execution with an ``if __name__ == '__main__':``
            statement when working on Windows.
            Currently, the use of an HDF5Cache is not supported in parallel on Windows
            platforms. This is due to the way subprocesses are forked in this
            architecture. The method
            :meth:`.DOEScenario.set_optimization_history_backup` is recommended as
            an alternative.
        """  # noqa: D205, D212, D415
        self.__hdf_node_path = hdf_node_path
        self.__hdf_file = HDF5FileSingleton(str(hdf_file_path))
        super().__init__(tolerance, name or hdf_node_path)
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
    def _string_representation(self) -> MultiLineString:
        mls = super()._string_representation
        mls.add("HDF file path: {}", self.__hdf_file.hdf_file_path)
        mls.add("HDF node path: {}", self.__hdf_node_path)
        return mls

    def __getstate__(self) -> dict[str, float | str]:
        # Pickle __init__ arguments so to call it when unpickling.
        return {
            "tolerance": self.tolerance,
            "hdf_file_path": self.__hdf_file.hdf_file_path,
            "hdf_node_path": self.__hdf_node_path,
            "name": self.name,
        }

    def __setstate__(self, state: StrKeyMapping) -> None:
        self.__class__.__init__(self, **state)

    def _copy_empty_cache(self) -> HDF5Cache:
        file_path = Path(self.__hdf_file.hdf_file_path)
        return self.__class__(
            hdf_file_path=file_path.parent / ("new_" + file_path.name),
            hdf_node_path=self.__hdf_node_path,
            tolerance=self.tolerance,
            name=self.name,
        )

    def _set_lock(self) -> RLockType:
        return self.__hdf_file.lock

    @synchronized
    def _read_hashes(self) -> None:
        """Read the hashes dict in the HDF file."""
        max_index = self.__hdf_file.read_hashes(
            self._hashes_to_indices, self.__hdf_node_path
        )
        self._last_accessed_index.value = max_index
        self._max_index.value = max_index
        cache_size = len(self._hashes_to_indices)
        if cache_size > 0:
            msg = "Found %s entries in the cache file : %s node : %s"
            LOGGER.info(
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
        if group == self.Group.JACOBIAN and data:
            data = nest_flat_bilevel_dict(data, separator=self._JACOBIAN_SEPARATOR)
        return data

    def _write_data(
        self,
        values: StrKeyMapping,
        group: BaseFullCache.Group,
        index: int,
    ) -> None:
        self.__hdf_file.write_data(
            values,
            group,
            index,
            self.__hdf_node_path,
        )

    @synchronized
    def get_all_entries(self) -> Iterator[CacheEntry]:  # noqa: D102
        with self.__hdf_file.keep_open():
            for index in self._all_groups:
                input_data = self._read_data(index, self.Group.INPUTS)
                output_data = self._read_data(index, self.Group.OUTPUTS)
                jacobian_data = self._read_data(index, self.Group.JACOBIAN)
                yield CacheEntry(input_data, output_data, jacobian_data)

    @staticmethod
    def update_file_format(
        hdf_file_path: str | Path,
    ) -> None:
        """Update the format of a HDF5 file.

        .. seealso:: :meth:`.HDF5FileSingleton.update_file_format`.

        Args:
            hdf_file_path: A HDF5 file path.
        """
        HDF5FileSingleton.update_file_format(hdf_file_path)
