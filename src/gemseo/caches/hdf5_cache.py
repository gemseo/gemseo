# -*- coding: utf-8 -*-
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
"""Caching module to store all the entries in a HDF file."""
from __future__ import division
from __future__ import unicode_literals

import logging
from typing import Any
from typing import Generator
from typing import Optional
from typing import Tuple
from typing import Union

import h5py

from gemseo.caches.hdf5_file_singleton import HDF5FileSingleton
from gemseo.core.cache import AbstractFullCache
from gemseo.core.cache import CacheEntry
from gemseo.core.cache import Data
from gemseo.core.cache import JacobianData
from gemseo.core.cache import OutputData
from gemseo.utils.data_conversion import nest_flat_bilevel_dict
from gemseo.utils.locks import synchronized
from gemseo.utils.multi_processing import RLock
from gemseo.utils.py23_compat import Path
from gemseo.utils.string_tools import MultiLineString

LOGGER = logging.getLogger(__name__)


class HDF5Cache(AbstractFullCache):
    """Cache using disk HDF5 file to store the data."""

    def __init__(
        self,
        hdf_file_path="cache.hdf5",  # type: Union[str, Path]
        hdf_node_path="node",  # type: str
        tolerance=0.0,  # type: float
        name=None,  # type: Optional[str]
    ):  # type: (...) -> None
        """
        Args:
            hdf_file_path: The path of the HDF file.
                Initialize a singleton to access the HDF file.
                This singleton is used for multithreaded/multiprocessing access
                with a lock.
            hdf_node_path: The node of the HDF file.
            name: A name for the cache.
                If ``None``, use ``hdf_note_path``.
        """
        self.__hdf_node_path = hdf_node_path
        self.__hdf_file = HDF5FileSingleton(str(hdf_file_path))
        if not name:
            name = hdf_node_path

        super(HDF5Cache, self).__init__(tolerance, name)
        self._read_hashes()

    def __str__(self):  # type: (...) -> str
        msg = MultiLineString()
        msg.add(super(HDF5Cache, self).__str__())
        msg.indent()
        msg.add("HDF file path: {}", self.__hdf_file.hdf_file_path)
        msg.add("HDF node path: {}", self.__hdf_node_path)
        return str(msg)

    def __getstate__(self):
        # Pickle __init__ arguments so to call it when unpickling.
        return dict(
            tolerance=self.tolerance,
            hdf_file_path=self.__hdf_file.hdf_file_path,
            hdf_node_path=self.__hdf_node_path,
            name=self.name,
        )

    def __setstate__(self, state):
        self.__init__(**state)

    def _copy_empty_cache(self):  # type: (...) -> HDF5Cache
        file_path = Path(self.__hdf_file.hdf_file_path)
        return self.__class__(
            hdf_file_path=file_path.parent / ("new_" + file_path.name),
            hdf_node_path=self.__hdf_node_path,
            tolerance=self.tolerance,
            name=self.name,
        )

    def _set_lock(self):  # type: (...) -> RLock
        return self.__hdf_file.lock

    @synchronized
    def _read_hashes(self):  # type: (...) -> None
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
        index,  # type: int
        group,  # type: str
    ):  # type: (...) -> bool
        return self.__hdf_file.has_group(index, group, self.__hdf_node_path)

    @synchronized
    def clear(self):  # type: (...) -> None
        super(HDF5Cache, self).clear()
        self.__hdf_file.clear(self.__hdf_node_path)

    def _read_data(
        self,
        index,  # type: int
        group,  # type: str
        h5_open_file=None,  # type: Optional[h5py.File]
        **options,  # type: Any
    ):  # type: (...) -> Tuple[OutputData, JacobianData]
        """
        Args:
            h5_open_file: The opened HDF file.
                This improves performance
                but is incompatible with multiprocess/treading.
                If ``None``, open it.
        """
        data = self.__hdf_file.read_data(
            index, group, self.__hdf_node_path, h5_open_file=h5_open_file
        )[0]
        if group == self._JACOBIAN_GROUP and data is not None:
            data = nest_flat_bilevel_dict(data, separator=self._JACOBIAN_SEPARATOR)

        return data

    def _write_data(
        self,
        data,  # type: Data
        group,  # type: str
        index,  # type: int
    ):  # type: (...) -> None
        self.__hdf_file.write_data(
            data,
            group,
            index,
            self.__hdf_node_path,
        )

    @synchronized
    def __iter__(
        self,
    ):  # type: (...) -> Generator[CacheEntry]
        with h5py.File(self.__hdf_file.hdf_file_path, "a") as h5_open_file:
            for data in self._all_data(h5_open_file=h5_open_file):
                yield data

    @staticmethod
    def update_file_format(
        hdf_file_path,  # type: Union[str, Path]
    ):  # type: (...) -> None
        """Update the format of a HDF5 file.

        .. seealso:: :meth:`.HDF5FileSingleton.update_file_format`.

        Args:
            hdf_file_path: A HDF5 file path.
        """
        HDF5FileSingleton.update_file_format(hdf_file_path)
