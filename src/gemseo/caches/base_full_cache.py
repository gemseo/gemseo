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
"""A base class for caches storing all data."""

from __future__ import annotations

from abc import abstractmethod
from itertools import chain
from multiprocessing import RLock
from multiprocessing import Value
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Literal
from typing import cast
from typing import overload

from numpy import append
from numpy import array
from numpy import concatenate
from numpy import vstack

from gemseo.caches.base_cache import BaseCache
from gemseo.caches.cache_entry import CacheEntry
from gemseo.caches.utils import hash_data
from gemseo.utils.data_conversion import flatten_nested_bilevel_dict
from gemseo.utils.ggobi_export import save_data_arrays_to_xml
from gemseo.utils.locks import synchronized
from gemseo.utils.locks import synchronized_hashes
from gemseo.utils.multiprocessing.manager import get_multi_processing_manager

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Iterator
    from multiprocessing.managers import DictProxy
    from multiprocessing.sharedctypes import Synchronized
    from multiprocessing.synchronize import RLock as RLockType

    from gemseo.typing import IntegerArray
    from gemseo.typing import JacobianData
    from gemseo.typing import StrKeyMapping


class BaseFullCache(BaseCache):
    """Base cache to store all the data, either in memory or on the disk.

    See Also:
        :class:`.MemoryFullCache`: store all the data in memory.
        :class:`.HDF5Cache`: store all the data in an HDF5 file.
    """

    _JACOBIAN_SEPARATOR: ClassVar[str] = "!d$_$d!"
    """The string separating the input and output names in a derivative name.

    E.g. ``"output!d$_$d!input"``.
    """

    lock: RLockType
    """The lock used for both multithreading and multiprocessing.

    Ensure safe multiprocessing and multithreading concurrent access to the cache.
    """

    lock_hashes: RLockType
    """The lock used for both multithreading and multiprocessing.

    Ensure safe multiprocessing and multithreading concurrent access to the cache.
    """

    _hashes_to_indices: DictProxy[int, IntegerArray]
    """The indices associated with the hashes."""

    _max_index: Synchronized[int]
    """The maximum index of the data stored in the cache."""

    _last_accessed_index: Synchronized[int]
    """The index of the last accessed data."""

    def __init__(  # noqa: D107
        self,
        tolerance: float = 0.0,
        name: str = "",
    ) -> None:
        super().__init__(tolerance, name)
        self.lock_hashes = RLock()
        self._hashes_to_indices = get_multi_processing_manager().dict()
        self._max_index = cast("Synchronized[int]", Value("i", 0))
        self._last_accessed_index = cast("Synchronized[int]", Value("i", 0))
        self.lock = self._set_lock()

    @abstractmethod
    def _set_lock(self) -> RLockType:
        """Set a lock for multithreading.

        Either from an external object or internally by using RLock().
        """

    def __ensure_input_data_exists(
        self,
        input_data: StrKeyMapping,
    ) -> bool:
        """Ensure ``input_data`` associated with ``data_hash`` exists.

        If ``input_data`` is cached,
        return ``True``.
        If ``data_hash`` is missing,
        store this hash and index ``input_data`` before caching later at this index.
        If ``data_hash`` exists but ``input_data`` is not cached,
        add ``data_hash`` and then index ``input_data``.

        Args:
            input_data: The input data to cache.

        Returns:
            Whether ``input_data`` was missing.
        """
        data_hash = hash_data(input_data)

        # Check if there is an entry with this hash in the cache.
        indices = self._hashes_to_indices.get(data_hash)

        # If no, initialize a new entry.
        if indices is None:
            self._max_index.value += 1
            self._last_accessed_index.value = self._max_index.value
            self._hashes_to_indices[data_hash] = array([self._max_index.value])
            self._initialize_entry(self._max_index.value)
            return True

        # If yes, look if there is a corresponding input data equal to ``input_data``.
        for index in indices:
            if self.compare_dict_of_arrays(
                input_data, self._read_data(index, self.Group.INPUTS)
            ):
                # The input data is already cached => we don't store it again.
                self._last_accessed_index.value = index
                return False

        # If there is no an input data equal ``input_data``,
        # update the indices related to the ``data_hash``.
        self._max_index.value += 1
        self._last_accessed_index.value = self._max_index.value
        self._hashes_to_indices[data_hash] = append(indices, self._max_index.value)
        self._initialize_entry(self._max_index.value)
        return True

    def _initialize_entry(
        self,
        index: int,
    ) -> None:
        """Initialize an entry of the cache if needed.

        Args:
            index: The index of the entry.
        """

    @abstractmethod
    def _has_group(
        self,
        index: int,
        group: BaseCache.Group,
    ) -> bool:
        """Check if an entry has data corresponding to a given group.

        Args:
            index: The index of the entry.
            group: The group.

        Returns:
            Whether the entry has data for this group.
        """

    @abstractmethod
    def _write_data(
        self,
        values: StrKeyMapping,
        group: BaseCache.Group,
        index: int,
    ) -> None:
        """Write the data associated with an index and a group.

        Args:
            values: The data containing the values of the names to cache.
            group: The group.
            index: The index of the entry in the cache.
        """

    def _cache_inputs(
        self,
        input_data: StrKeyMapping,
        group: BaseCache.Group,
    ) -> bool:
        """Cache input data and increment group if needed.

        Cache inputs and increment group if needed.
        Check if ``group`` exists for these inputs.

        This method avoids duplicate storage.

        Args:
            input_data: The data containing the input data to cache.
            group: The group.

        Returns:
            Whether ``group`` exists.
        """
        if self.__ensure_input_data_exists(input_data):
            self._write_data(input_data, self.Group.INPUTS, self._max_index.value)
        elif self._has_group(self._last_accessed_index.value, group):
            return True
        return False

    @synchronized
    def cache_outputs(  # noqa: D102
        self,
        input_data: StrKeyMapping,
        output_data: StrKeyMapping,
    ) -> None:
        if self._cache_inputs(input_data, self.Group.OUTPUTS):
            # There is already an output data corresponding to this input data.
            return

        self._write_data(
            output_data,
            self.Group.OUTPUTS,
            self._last_accessed_index.value,
        )

    @synchronized
    def cache_jacobian(  # noqa: D102
        self,
        input_data: StrKeyMapping,
        jacobian_data: JacobianData,
    ) -> None:
        if self._cache_inputs(input_data, self.Group.JACOBIAN):
            # There is already a Jacobian data corresponding to this input data.
            return

        flat_jacobian_data = flatten_nested_bilevel_dict(
            jacobian_data, separator=self._JACOBIAN_SEPARATOR
        )

        self._write_data(
            flat_jacobian_data,
            self.Group.JACOBIAN,
            self._last_accessed_index.value,
        )

    @synchronized
    def clear(self) -> None:  # noqa: D102
        super().clear()
        self._hashes_to_indices.clear()
        self._max_index.value = 0
        self._last_accessed_index.value = 0

    @property
    @synchronized
    def last_entry(self) -> CacheEntry:  # noqa: D102
        if not self:
            return CacheEntry({}, {}, {})

        return CacheEntry(
            self._read_data(self._last_accessed_index.value, self.Group.INPUTS),
            self._read_data(self._last_accessed_index.value, self.Group.OUTPUTS),
            self._read_data(self._last_accessed_index.value, self.Group.JACOBIAN),
        )

    @synchronized
    def __len__(self) -> int:
        return self._max_index.value

    @overload
    def _read_data(
        self,
        index: int,
        group: Literal[BaseCache.Group.INPUTS, BaseCache.Group.OUTPUTS],
    ) -> StrKeyMapping: ...

    @overload
    def _read_data(
        self,
        index: int,
        group: Literal[BaseCache.Group.JACOBIAN],
    ) -> JacobianData: ...

    @abstractmethod
    def _read_data(
        self,
        index: int,
        group: BaseCache.Group,
    ) -> StrKeyMapping | JacobianData:
        """Read the data of an entry.

        Args:
            index: The index of the entry.
            group: The group.

        Returns:
            The output and Jacobian data corresponding to these index and group.
        """

    @synchronized_hashes
    def __has_hash(
        self,
        data_hash: int,
    ) -> IntegerArray | None:
        """Get the indices corresponding to a data hash.

        Args:
            The data hash.

        Returns:
            The indices corresponding to this data hash.
        """
        return self._hashes_to_indices.get(data_hash)

    def _read_input_output_data(
        self,
        indices: Iterable[int],
        input_data: StrKeyMapping,
    ) -> CacheEntry:
        """Read the output and Jacobian data for a given input data.

        Args:
            indices: The indices of the entries among from which the entry to read data.
            input_data: The input data.

        Returns:
            The output and Jacobian data if they exist, ``None`` otherwise.
        """
        for index in indices:
            if self.compare_dict_of_arrays(
                input_data, self._read_data(index, self.Group.INPUTS)
            ):
                output_data = self._read_data(index, self.Group.OUTPUTS)
                jacobian_data = self._read_data(index, self.Group.JACOBIAN)
                return CacheEntry(input_data, output_data, jacobian_data)

        return CacheEntry(input_data, {}, {})

    @synchronized
    def __getitem__(
        self,
        input_data: StrKeyMapping,
    ) -> CacheEntry:
        if self._tolerance == 0.0:
            data_hash = hash_data(input_data)
            indices = self.__has_hash(data_hash)
            if indices is None:
                return CacheEntry(input_data, {}, {})

            return self._read_input_output_data(indices, input_data)

        for indices in self._hashes_to_indices.values():
            for index in indices:
                cached_input_data = self._read_data(index, self.Group.INPUTS)
                if self.compare_dict_of_arrays(
                    input_data, cached_input_data, self._tolerance
                ):
                    output_data = self._read_data(index, self.Group.OUTPUTS)
                    jacobian_data = self._read_data(index, self.Group.JACOBIAN)
                    return CacheEntry(input_data, output_data, jacobian_data)

        return CacheEntry(input_data, {}, {})

    @property
    def _all_groups(self) -> list[int]:
        """Sorted the indices of the entries."""
        return sorted(chain(*(v.tolist() for v in self._hashes_to_indices.values())))

    @synchronized
    def get_all_entries(self) -> Iterator[CacheEntry]:  # noqa: D102
        for index in self._all_groups:
            input_data = self._read_data(index, self.Group.INPUTS)
            output_data = self._read_data(index, self.Group.OUTPUTS)
            jacobian_data = self._read_data(index, self.Group.JACOBIAN)
            yield CacheEntry(input_data, output_data, jacobian_data)

    def to_ggobi(
        self,
        file_path: str,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        """Export the cache to an XML file for ggobi tool.

        Args:
            file_path: The path of the file to export the cache.
            input_names: The names of the inputs to export.
                If empty, export all of them.
            output_names: The names of the outputs to export.
                If empty, export all of them.
        """
        if not self._hashes_to_indices:
            msg = "An empty cache cannot be exported to XML file."
            raise ValueError(msg)

        shared_input_names: set[str] = set()
        shared_output_names: set[str] = set()
        all_input_data = []
        all_output_data = []
        names_to_sizes = {}

        for data in self.get_all_entries():
            input_data = data.inputs or {}
            output_data = data.outputs or {}
            try:
                if input_names:
                    input_data = {name: input_data[name] for name in input_names}

                if output_names:
                    output_data = {name: output_data[name] for name in output_names}

            except KeyError:
                # The data is not in this execution
                continue

            # Compute the size of the data
            names_to_sizes.update({key: val.size for key, val in input_data.items()})
            names_to_sizes.update({key: val.size for key, val in output_data.items()})
            current_input_names = set(input_data.keys())
            current_output_names = set(output_data.keys())
            shared_input_names = (
                shared_input_names or current_input_names
            ) & current_input_names
            shared_output_names = (
                shared_output_names or current_output_names
            ) & current_output_names
            all_input_data.append(input_data)
            all_output_data.append(output_data)

        if not all_output_data:
            msg = "Failed to find outputs in the cache."
            raise ValueError(msg)

        variable_names = []
        for data_name in list(shared_input_names) + list(shared_output_names):
            data_size = names_to_sizes[data_name]
            if data_size == 1:
                variable_names.append(data_name)
            else:
                variable_names += [f"{data_name}_{i + 1}" for i in range(data_size)]

        cache_as_array = vstack([
            concatenate(
                [all_input_data[index][name].flatten() for name in shared_input_names]
                + [
                    all_output_data[index][name].flatten()
                    for name in shared_output_names
                ]
            )
            for index in range(len(all_input_data))
        ])
        save_data_arrays_to_xml(variable_names, cache_as_array, file_path)

    def update(
        self,
        other_cache: BaseFullCache,
    ) -> None:
        """Update from another cache.

        Args:
            other_cache: The cache to update the current one.
        """
        for input_data, output_data, jacobian_data in other_cache.get_all_entries():
            if output_data or jacobian_data:
                self[input_data] = (output_data, jacobian_data)

    @abstractmethod
    def _copy_empty_cache(self) -> BaseFullCache:
        """Copy a cache without its entries."""

    def __add__(
        self,
        other_cache: BaseFullCache,
    ) -> BaseFullCache:
        """Concatenate a cache to a copy of the current one.

        Args:
            other_cache: A cache to be concatenated to a copy of the current one.

        Returns:
            A new cache concatenating the current one and ``other_cache``.
        """
        new_cache = self._copy_empty_cache()
        new_cache.update(self)
        new_cache.update(other_cache)
        return new_cache
