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

"""HDF5 file singleton used by the HDF5 cache."""
from typing import ClassVar, Dict, Optional, Union

import h5py
from genericpath import exists
from numpy import append, bytes_, ndarray, unicode_
from numpy.core.multiarray import array
from six import PY2, with_metaclass

from gemseo.core.cache import AbstractFullCache, Data, hash_data_dict, to_real
from gemseo.utils.multi_processing import RLock
from gemseo.utils.py23_compat import Path, long, string_array, string_dtype
from gemseo.utils.singleton import SingleInstancePerFileAttribute


class HDF5FileSingleton(with_metaclass(SingleInstancePerFileAttribute, object)):
    """Singleton to access a HDF file.

    Used for multithreaded/multiprocessing access with a lock.
    """

    # We create a single instance of cache per HDF5 file
    # to allow the multiprocessing lock to work
    # Ie we want a single lock even if we instantiate multiple
    # caches

    hdf_file_path: str
    """The path to the HDF file."""

    lock: RLock
    """The lock used for multithreading."""

    HASH_TAG: ClassVar[str] = "hash"
    """The label for the hash."""

    FILE_FORMAT_VERSION: ClassVar[int] = 2
    """The version of the file format."""

    _INPUTS_GROUP: ClassVar[str] = AbstractFullCache._INPUTS_GROUP
    """The label for the input variables."""

    def __init__(
        self,
        hdf_file_path,  # type: str
    ):  # type: (...) -> None
        """
        Args:
            hdf_file_path: The path to the HDF file.
        """
        self.hdf_file_path = hdf_file_path
        self.__check_file_format_version()
        # Attach the lock to the file and NOT the Cache because it is a singleton.
        self.lock = RLock()

    def write_data(
        self,
        data,  # type: Data
        group,  # type: str
        index,  # type: int
        hdf_node_path,  # type: str
        h5_open_file=None,  # type: Optional[h5py.File]
    ):  # type: (...) -> None
        """Cache input data to avoid re-evaluation.

        Args:
            data: The data containing the values of the names to cache.
            group: The name of the group,
                either :attr:`.AbstractFullCache._INPUTS_GROUP`,
                :attr:`.AbstractFullCache._OUTPUTS_GROUP`
                or :attr:`.AbstractFullCache._JACOBIAN_GROUP`.
            index: The index of the entry in the cache.
            hdf_node_path: The name of the HDF group to store the entries.
            h5_open_file: The opened HDF file.
                This improves performance
                but is incompatible with multiprocess/treading.
                If ``None``, open it.
        """
        if h5_open_file is None:
            h5_file = h5py.File(self.hdf_file_path, "a")
        else:
            h5_file = h5_open_file

        if not len(h5_file):
            self.__set_file_format_version(h5_file)

        root = h5_file.require_group(hdf_node_path)
        entry = root.require_group(str(index))
        group = entry.require_group(group)

        try:
            # Write hash if needed
            if entry.get(self.HASH_TAG) is None:
                data_hash = string_array([hash_data_dict(data)])
                entry.create_dataset(self.HASH_TAG, data=data_hash)

            for name, value in data.items():
                value = data.get(name)
                if value is not None:
                    if value.dtype.type is unicode_:
                        group.create_dataset(name, data=data.astype(string_dtype))
                    else:
                        group.create_dataset(name, data=to_real(value))

        # IOError and RuntimeError are for python 2.7
        except (RuntimeError, IOError, ValueError):
            h5_file.close()
            raise RuntimeError(
                "Failed to cache dataset %s.%s.%s in file: %s",
                hdf_node_path,
                index,
                group,
                self.hdf_file_path,
            )

        if h5_open_file is None:
            h5_file.close()

    def read_data(
        self,
        index,  # type: int
        group,  # type: str
        hdf_node_path,  # type: str
        h5_open_file=None,  # type: Optional[h5py.File]
    ):  # type: (...) -> Union[Optional[Data],Optional[int]]
        """Read the data for given index and group.

        Args:
            index:  The index of the entry.
            group: The name of the group.
            hdf_node_path: The name of the HDF group where the entries are stored.
            h5_open_file: The opened HDF file.
                This improves performance
                but is incompatible with multiprocess/treading.
                If ``None``, open it.

        Returns:
            The group data and the input data hash.
        """
        if h5_open_file is None:
            h5_file = h5py.File(self.hdf_file_path, "r")
        else:
            h5_file = h5_open_file

        root = h5_file[hdf_node_path]

        if not self._has_group(index, group, hdf_node_path, h5_file):
            return None, None

        entry = root[str(index)]
        data = {key: array(val) for key, val in entry[group].items()}
        if group == self._INPUTS_GROUP:
            hash_ = entry[self.HASH_TAG]
            hash_ = long(array(hash_)[0])
        else:
            hash_ = None

        if h5_open_file is None:
            h5_file.close()
        ##########################################
        # Python  key.encode("ascii")
        ##########################################
        if PY2:
            data = {name.encode("ascii"): value for name, value in data.items()}

        for name, value in data.items():
            if value.dtype.type is bytes_:
                data[name] = value.astype(unicode_)

        return data, hash_

    @staticmethod
    def _has_group(
        index,  # type: int
        group,  # type: str
        hdf_node_path,  # type: str
        h5_open_file,  # type: h5py.File
    ):  # type: (...) -> bool
        """
        Args:
            hdf_node_path: The name of the HDF group where the entries are stored.
            h5_open_file: The opened HDF file.
                This improves performance
                but is incompatible with multiprocess/treading.
        """
        entry = h5_open_file[hdf_node_path].get(str(index))
        if entry is None:
            return False

        if entry.get(group) is None:
            return False

        return True

    def has_group(
        self,
        index,  # type: int
        group,  # type: str
        hdf_node_path,  # type: str
    ):  # type: (...) -> bool
        """Check if an entry has data corresponding to a given group.

        Args:
            index:  The index of the entry.
            group: The name of the group.
            hdf_node_path: The name of the HDF group where the entries are stored.

        Returns:
            Whether the entry has data for this group.
        """
        with h5py.File(self.hdf_file_path, "r") as h5file:
            return self._has_group(index, group, hdf_node_path, h5file)

    def read_hashes(
        self,
        hashes_to_indices,  # type: Dict[str,ndarray]
        hdf_node_path,  # type: str
    ):  # type: (...) -> int
        """Read the hashes in the HDF file.

        Args:
            hashes_to_indices: The indices associated to the hashes.
            hdf_node_path: The name of the HDF group where the entries are stored.

        Returns:
            The maximum index.
        """
        if not exists(self.hdf_file_path):
            return 0

        # We must lock so that no data is added to the cache meanwhile
        with h5py.File(self.hdf_file_path, "r") as h5file:
            root = h5file.get(hdf_node_path)

            if root is None:
                return 0

            max_index = 0

            for index, entry in root.items():
                index = int(index)
                hash_ = long(array(entry[self.HASH_TAG])[0])
                indices = hashes_to_indices.get(hash_)

                if indices is None:
                    hashes_to_indices[hash_] = array([index])
                else:
                    hashes_to_indices[hash_] = append(indices, array([index]))

                max_index = max(max_index, index)

        return max_index

    def clear(
        self,
        hdf_node_path,  # type: str
    ):  # type: (...) -> None
        """
        Args:
            hdf_node_path: The name of the HDF group to clear.
        """
        with h5py.File(self.hdf_file_path, "a") as h5file:
            del h5file[hdf_node_path]

    def __check_file_format_version(self):  # type: (...) -> None
        """Make sure the file can be handled.

        Raises
            ValueError: If the version of the file format is missing or
                greater than the current one.
        """
        if not Path(self.hdf_file_path).exists():
            return

        h5_file = h5py.File(self.hdf_file_path, "r")

        if not len(h5_file):
            h5_file.close()
            return

        version = h5_file.attrs.get("version")
        h5_file.close()

        if version is None:
            raise ValueError(
                "The file {} cannot be used because it has no file format version: "
                "see HDFCache.update_file_format to convert it.".format(
                    self.hdf_file_path
                )
            )

        if version > self.FILE_FORMAT_VERSION:
            raise ValueError(
                "The file {} cannot be used because its file format version is {} "
                "while the expected version is {}: "
                "see HDFCache.update_file_format to convert it.".format(
                    self.hdf_file_path, version, self.FILE_FORMAT_VERSION
                )
            )

    @classmethod
    def __set_file_format_version(
        cls,
        h5_file,  # type: h5py.File
    ):  # type: (...) -> None
        """Change the version of an HDF5 file to :attr:`.FILE_FORMAT_VERSION`.

        Args:
            h5_file: A HDF5 file object.
        """
        h5_file.attrs["version"] = cls.FILE_FORMAT_VERSION

    @classmethod
    def update_file_format(
        cls,
        hdf_file_path,  # type: Union[str, Path]
    ):  # type: (...) -> None
        """Update the format of a HDF5 file.

        |g| 3.2.0 added a :attr:`.HDF5FileSingleton.FILE_FORMAT_VERSION`
        to the HDF5 files,
        to allow handling its maintenance and evolutions.
        In particular,
        |g| 3.2.0 fixed the hashing of the data dictionaries.

        Args:
            hdf_file_path: A HDF5 file path.
        """
        with h5py.File(str(hdf_file_path), "a") as h5file:
            cls.__set_file_format_version(h5file)
            for value in h5file.values():
                if not isinstance(value, h5py.Group):
                    continue

                for sample_value in value.values():
                    data = sample_value[cls._INPUTS_GROUP]
                    data = {key: array(val) for key, val in data.items()}
                    data_hash = string_array([hash_data_dict(data)])
                    sample_value[cls.HASH_TAG][0] = data_hash
