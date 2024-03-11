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

from contextlib import contextmanager
from multiprocessing import RLock
from pathlib import Path
from typing import TYPE_CHECKING
from typing import ClassVar

import h5py
from genericpath import exists
from h5py import File
from numpy import append
from numpy import bytes_
from numpy import str_
from numpy.core.multiarray import array
from scipy.sparse import csr_array
from strenum import StrEnum

from gemseo.core.cache import AbstractCache
from gemseo.core.cache import hash_data_dict
from gemseo.core.cache import to_real
from gemseo.utils.compatibility.scipy import SparseArrayType
from gemseo.utils.compatibility.scipy import sparse_classes
from gemseo.utils.singleton import SingleInstancePerFileAttribute

if TYPE_CHECKING:
    from collections.abc import Iterator
    from multiprocessing.managers import DictProxy
    from multiprocessing.synchronize import RLock as RLockType

    from gemseo.typing import DataMapping
    from gemseo.typing import IntegerArray


# TODO: API: make this module and class protected.
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

    lock: RLockType
    """The lock used for multithreading."""

    HASH_TAG: ClassVar[str] = "hash"
    """The label for the hash."""

    FILE_FORMAT_VERSION: ClassVar[int] = 2
    """The version of the file format."""

    __keep_open: bool
    """Whether to keep the file open when leaving :meth:`.__open` context manager."""

    __file: File | None
    """The hdf5 file handle."""

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
        self.__keep_open = False
        self.__file = None
        self.hdf_file_path = hdf_file_path
        self.__check_file_format_version()
        # Attach the lock to the file and NOT the Cache because it is a singleton.
        self.lock = RLock()

    def write_data(
        self,
        data: DataMapping,
        group: AbstractCache.Group,
        index: int,
        hdf_node_path: str,
    ) -> None:
        """Cache input data to avoid re-evaluation.

        Args:
            data: The data containing the values of the names to cache.
            group: The group.
            index: The index of the entry in the cache.
            hdf_node_path: The name of the HDF group to store the entries,
                possibly passed as a path ``root_name/.../group_name/.../node_name``.
        """
        with self.__open(mode="a"):
            assert self.__file is not None

            if not len(self.__file):
                self.__set_file_format_version(self.__file)

            root = self.__file.require_group(hdf_node_path)
            entry = root.require_group(str(index))
            entry_group = entry.require_group(group)

            try:
                # Write hash if needed
                if entry.get(self.HASH_TAG) is None:
                    data_hash = array([hash_data_dict(data)], dtype="bytes")
                    entry.create_dataset(self.HASH_TAG, data=data_hash)

                for name, value in data.items():
                    value = data.get(name)
                    if value is not None:
                        if value.dtype.type is str_:
                            entry_group.create_dataset(name, data=value.astype("bytes"))
                        elif isinstance(value, sparse_classes):
                            self.__write_sparse_array(entry_group, name, value)
                        else:
                            entry_group.create_dataset(name, data=to_real(value))

            # IOError and RuntimeError are for python 2.7
            except (RuntimeError, OSError, ValueError):
                msg = "Failed to cache dataset %s.%s.%s in file: %s"
                raise RuntimeError(
                    msg,
                    hdf_node_path,
                    index,
                    entry_group,
                    self.hdf_file_path,
                ) from None

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
        group: AbstractCache.Group,
        hdf_node_path: str,
    ) -> DataMapping:
        """Read the data for given index and group.

        Args:
            index:  The index of the entry.
            group: The group.
            hdf_node_path: The name of the HDF group where the entries are stored,
                possibly passed as a path ``root_name/.../group_name/.../node_name``.

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
                    data[key] = self.__read_sparse_array(dataset)
                else:
                    data[key] = array(dataset)

            for name, value in data.items():
                if value.dtype.type is bytes_:
                    data[name] = value.astype(str_)

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
        group: AbstractCache.Group,
        hdf_node_path: str,
    ) -> bool:
        """Return whether a group exists.

        Args:
            index: The index of the entry.
            group: The group.
            hdf_node_path: The name of the HDF group where the entries are stored,
                possibly passed as a path ``root_name/.../group_name/.../node_name``.

        Returns:
            Whether a group exists.
        """
        assert self.__file is not None
        entry = self.__file[hdf_node_path].get(str(index))
        if entry is None:
            return False
        if entry.get(group) is None:
            return False
        return True

    def has_group(
        self,
        index: int,
        group: AbstractCache.Group,
        hdf_node_path: str,
    ) -> bool:
        """Check if an entry has data corresponding to a given group.

        Args:
            index: The index of the entry.
            group: The group.
            hdf_node_path: The name of the HDF group where the entries are stored,
                possibly passed as a path ``root_name/.../group_name/.../node_name``.

        Returns:
            Whether the entry has data for this group.
        """
        with self.__open():
            return self._has_group(index, group, hdf_node_path)

    def read_hashes(
        self,
        hashes_to_indices: DictProxy[int, IntegerArray],
        hdf_node_path: str,
    ) -> int:
        """Read the hashes in the HDF file.

        Args:
            hashes_to_indices: The indices associated to the hashes.
            hdf_node_path: The name of the HDF group where the entries are stored,
                possibly passed as a path ``root_name/.../group_name/.../node_name``.

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
                hash_ = int(array(entry[self.HASH_TAG])[0])
                indices = hashes_to_indices.get(hash_)

                if indices is None:
                    hashes_to_indices[hash_] = array([index])
                else:
                    hashes_to_indices[hash_] = append(indices, array([index]))

                max_index = max(max_index, index)

        return max_index

    def clear(
        self,
        hdf_node_path: str,
    ) -> None:
        """Clear a node.

        Args:
            hdf_node_path: The name of the HDF group to clear,
                possibly passed as a path ``root_name/.../group_name/.../node_name``.
        """
        with self.__open(mode="a"):
            assert self.__file is not None
            del self.__file[hdf_node_path]

    def __check_file_format_version(self) -> None:
        """Make sure the file can be handled.

        Raises:
            ValueError: If the version of the file format is missing or
                greater than the current one.
        """
        if not Path(self.hdf_file_path).exists():
            return

        with self.__open():
            assert self.__file is not None
            if not len(self.__file):
                return
            version = self.__file.attrs.get("version")

        if version is None:
            msg = (
                f"The file {self.hdf_file_path} cannot be used because it has no file "
                "format version: see HDFCache.update_file_format to convert it."
            )
            raise ValueError(msg)

        if version > self.FILE_FORMAT_VERSION:
            msg = (
                f"The file {self.hdf_file_path} cannot be used because its file "
                f"format version is {version}"
                f"while the expected version is {self.FILE_FORMAT_VERSION}: "
                "see HDFCache.update_file_format to convert it."
            )
            raise ValueError(msg)

    @classmethod
    def __set_file_format_version(
        cls,
        h5_file: h5py.File,
    ) -> None:
        """Change the version of an HDF5 file to :attr:`.FILE_FORMAT_VERSION`.

        Args:
            h5_file: A HDF5 file object.
        """
        h5_file.attrs["version"] = cls.FILE_FORMAT_VERSION

    @classmethod
    def update_file_format(
        cls,
        hdf_file_path: str | Path,
    ) -> None:
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
                if isinstance(value, h5py.Group):
                    for sample_value in value.values():
                        data = sample_value[AbstractCache.Group.INPUTS]
                        data = {key: array(val) for key, val in data.items()}
                        data_hash = array([hash_data_dict(data)], dtype="bytes")
                        sample_value[cls.HASH_TAG][0] = data_hash

    @contextmanager
    def __open(self, mode: str = "r") -> Iterator[None]:
        """Open a hdf5 file.

        The file handle is used via :attr:`.__file`.

        Args:
            mode: The opening mode, see :meth:`h5py.File`.
        """
        if self.__keep_open and self.__file is not None:
            yield
        else:
            self.__file = h5py.File(self.hdf_file_path, mode=mode)
            yield
            if not self.__keep_open:
                self.__close()

    @contextmanager
    def keep_open(self) -> Iterator[None]:
        """Keep the file open for all file operations done in this context manager."""
        self.__keep_open = True
        yield
        self.__keep_open = False
        self.__close()

    def __close(self) -> None:
        """Close the file handle."""
        assert self.__file is not None
        self.__file.close()
        self.__file = None
