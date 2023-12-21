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
"""Caching module to avoid multiple evaluations of a discipline."""

from __future__ import annotations

import logging
import sys
from abc import abstractmethod
from collections.abc import Generator
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Mapping as ABCMapping
from collections.abc import Sized
from itertools import chain
from multiprocessing import RLock
from multiprocessing import Value
from typing import TYPE_CHECKING
from typing import Callable
from typing import ClassVar
from typing import NamedTuple

from numpy import append
from numpy import array
from numpy import ascontiguousarray
from numpy import complex128
from numpy import concatenate
from numpy import float64
from numpy import hstack
from numpy import int32
from numpy import int64
from numpy import ndarray
from numpy import uint8
from numpy import vstack
from pandas import MultiIndex
from xxhash import xxh3_64_hexdigest

from gemseo.datasets.dataset import Dataset
from gemseo.datasets.io_dataset import IODataset
from gemseo.utils.comparisons import DataToCompare
from gemseo.utils.comparisons import compare_dict_of_arrays
from gemseo.utils.data_conversion import flatten_nested_bilevel_dict
from gemseo.utils.ggobi_export import save_data_arrays_to_xml
from gemseo.utils.locks import synchronized
from gemseo.utils.locks import synchronized_hashes
from gemseo.utils.multiprocessing import get_multi_processing_manager
from gemseo.utils.string_tools import MultiLineString

if TYPE_CHECKING:
    from gemseo.core.discipline_data import Data

LOGGER = logging.getLogger(__name__)

JacobianData = Mapping[str, Mapping[str, ndarray]]

DATA_COMPARATOR: Callable[[DataToCompare, DataToCompare], bool] = compare_dict_of_arrays
"""The comparator of input data structures.

It is used to check whether an input data has been cached in.
"""


class CacheEntry(NamedTuple):
    """An entry of a cache."""

    inputs: Mapping[str, ndarray]
    """The input data."""

    outputs: Mapping[str, ndarray]
    """The output data."""

    jacobian: Mapping[str, Mapping[str, ndarray]]
    """The Jacobian data."""


# TODO: API: rename to BaseCache
class AbstractCache(ABCMapping):
    """An abstract base class for caches with a dictionary-like interface.

    Caches are mainly used to store the :class:`.MDODiscipline` evaluations.

    A cache entry is defined by:

    - an input data
      in the form of a dictionary of objects associated with input names,
      i.e. ``{"input_name": object}``,
    - an output data
      in the form of a dictionary of NumPy arrays associated with output names,
      i.e. ``{"output_name": array}``.
    - an optional Jacobian data,
      in the form of a nested dictionary of NumPy arrays
      associating output and input names,
      i.e. ``{"output_name": {"input_name": array}}``.

    Examples:
        The evaluation of the function :math:`y=f(x)=(x^2, 2x^3`)`
        and its derivative at :math:`x=1` leads to cache the entry defined by:

        - the input data: :math:`1.`,
        - the output data: :math:`(1., 2.)`,
        - the Jacobian data: :math:`(2., 6.)^T`.

        >>> input_data = {"x": array([1.0])}
        >>> output_data = {"y": array([1.0, 2.0])}
        >>> jacobian_data = {"y": {"x": array([[2.0], [6.0]])}}

        For this ``input_data``,
        one can cache the output data:

        >>> cache.cache_outputs(input_data, output_data)

        as well as the Jacobian data:

        >>> cache.cache_jacobian(input_data, jacobian_data)

    Caches have a :class:`.abc.Mapping` interface
    making them easy to set (``cache[input_data] = (output_data, jacobian_data)``),
    access (``cache_entry = cache[input_data]``)
    and update (``cache.update(other_cache)``).

    Notes:
        ``cache_entry`` is a :class:`.CacheEntry`
        with the ordered fields *input*, *output* and *jacobian*
        accessible either by index, e.g. ``input_data = cache_entry[0]``,
        or by name, e.g. ``input_data = cache_entry.inputs``.

    Notes:
        If an output name is also an input name,
        the output name is suffixed with ``[out]``.

    One can also get the number of cache entries with ``size = len(cache)``
    and iterate over the cache,
    e.g. ``for input_data, output_data, _ in cache``
    ``for index, (input_data, _, jacobian_data) in enumerate(cache)``
    or ``[entry.outputs for entry in cache]``.

    See Also:
        :class:`.SimpleCache` to store the last discipline evaluation.
        :class:`.MemoryFullCache` to store all the discipline evaluations in memory.
        :class:`.HDF5Cache` to store all the discipline evaluations in a HDF5 file.
    """

    name: str
    """The name of the cache."""

    tolerance: float
    """The tolerance below which two input arrays are considered equal."""

    _INPUTS_GROUP: ClassVar[str] = "inputs"
    """The label for the input variables."""

    _OUTPUTS_GROUP: ClassVar[str] = "outputs"
    """The label for the output variables."""

    _JACOBIAN_GROUP: ClassVar[str] = "jacobian"
    """The label for the Jacobian."""

    def __init__(
        self,
        tolerance: float = 0.0,
        name: str | None = None,
    ) -> None:
        """
        Args:
            tolerance: The tolerance below which two input arrays are considered equal:
                ``norm(new_array-cached_array)/(1+norm(cached_array)) <= tolerance``.
                If this is the case for all the input names,
                then the cached output data shall be returned
                rather than re-evaluating the discipline.
                This tolerance could be useful to optimize CPU time.
                It could be something like ``2 * numpy.finfo(float).eps``.
            name: A name for the cache.
                If ``None``, use the class name.
        """  # noqa: D205, D212, D415
        self.tolerance = tolerance
        self.name = name if name is not None else self.__class__.__name__
        self.__names_to_sizes = {}
        self.__input_names = []
        self._output_names = []

    @property
    def input_names(self) -> list[str]:
        """The names of the inputs of the last entry."""
        if not self.__input_names:
            self.__input_names = sorted(self.last_entry.inputs.keys())
        return self.__input_names

    @property
    def output_names(self) -> list[str]:
        """The names of the outputs of the last entry."""
        if not self._output_names:
            self._output_names = sorted(self.last_entry.outputs.keys())
        return self._output_names

    @property
    def names_to_sizes(self) -> dict[str, int]:
        """The sizes of the variables of the last entry.

        For a Numpy array, its size is used. For a container, its length is used.
        Otherwise, a size of 1 is used.
        """
        if not self.__names_to_sizes:
            last_entry = self.last_entry
            for name, data in chain(
                last_entry.inputs.items(), last_entry.outputs.items()
            ):
                if isinstance(data, ndarray):
                    size = data.size
                elif isinstance(data, Sized):
                    size = len(data)
                else:
                    size = 1
                self.__names_to_sizes[name] = size

        return self.__names_to_sizes

    @property
    def _string_representation(self) -> MultiLineString:
        """The string representation of the cache."""
        mls = MultiLineString()
        mls.add("Name: {}", self.name)
        mls.indent()
        mls.add("Type: {}", self.__class__.__name__)
        mls.add("Tolerance: {}", self.tolerance)
        mls.add("Input names: {}", self.input_names)
        mls.add("Output names: {}", self.output_names)
        mls.add("Length: {}", len(self))
        return mls

    def __repr__(self) -> str:
        return str(self._string_representation)

    def _repr_html_(self) -> str:
        return self._string_representation._repr_html_()

    def __setitem__(
        self,
        input_data: Data,
        data: tuple[Data | None, JacobianData | None],
    ) -> None:
        output_data, jacobian_data = data
        if not output_data and not jacobian_data:
            LOGGER.warning(
                "Cannot add the entry to the cache "
                "as both output data and Jacobian data are missing."
            )
        if output_data:
            self.cache_outputs(input_data, output_data)

        if jacobian_data:
            self.cache_jacobian(input_data, jacobian_data)

    @abstractmethod
    def __getitem__(
        self,
        input_data: Data,
    ) -> CacheEntry: ...

    @abstractmethod
    def cache_outputs(
        self,
        input_data: Data,
        output_data: Data,
    ) -> None:
        """Cache input and output data.

        Args:
            input_data: The data containing the input data to cache.
            output_data: The data containing the output data to cache.
        """

    @abstractmethod
    def cache_jacobian(
        self,
        input_data: Data,
        jacobian_data: JacobianData,
    ) -> None:
        """Cache the input and Jacobian data.

        Args:
            input_data: The data containing the input data to cache.
            jacobian_data: The Jacobian data to cache.
        """

    def clear(self) -> None:
        """Clear the cache."""
        self.__names_to_sizes = {}
        self.__input_names = []
        self._output_names = []

    @property
    @abstractmethod
    def last_entry(self) -> CacheEntry:
        """The last cache entry."""

    def to_dataset(
        self,
        name: str = "",
        categorize: bool = True,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> Dataset:
        """Build a :class:`.Dataset` from the cache.

        Args:
            name: A name for the dataset.
                If empty, use the name of the cache.
            categorize: Whether to distinguish
                between the different groups of variables.
                Otherwise, group all the variables in :attr:`.Dataset.PARAMETER_GROUP``.
            input_names: The names of the inputs to be exported.
                If empty, use all the inputs.
            output_names: The names of the outputs to be exported.
                If empty, use all the outputs.
                If an output name is also an input name,
                the output name is suffixed with ``[out]``.

        Returns:
            A dataset version of the cache.
        """
        dataset_name = name or self.name
        if categorize:
            dataset_class = IODataset
            input_group = IODataset.INPUT_GROUP
            output_group = IODataset.OUTPUT_GROUP
        else:
            dataset_class = Dataset
            input_group = output_group = Dataset.DEFAULT_GROUP

        data = []
        columns = []
        for variable_names, group_name, is_output_group in zip(
            [input_names or self.input_names, output_names or self.output_names],
            [input_group, output_group],
            [False, True],
        ):
            for variable_name in variable_names:
                cache_entries = []
                for cache_entry in self:
                    if cache_entry.outputs:
                        if is_output_group:
                            selected_cache_entry = cache_entry.outputs[variable_name]
                        else:
                            selected_cache_entry = cache_entry.inputs[variable_name]
                        cache_entries.append(selected_cache_entry)

                new_data = vstack(cache_entries)
                data.append(new_data)
                columns.extend([
                    (group_name, variable_name, i) for i in range(new_data.shape[1])
                ])

        return dataset_class(
            hstack(data),
            dataset_name=dataset_name,
            columns=MultiIndex.from_tuples(
                columns,
                names=dataset_class.COLUMN_LEVEL_NAMES,
            ),
        )


# TODO: API: rename to BaseFullCache
class AbstractFullCache(AbstractCache):
    """Abstract cache to store all the data, either in memory or on the disk.

    See Also:
        :class:`.MemoryFullCache`: store all the data in memory.
        :class:`.HDF5Cache`: store all the data in an HDF5 file.
    """

    _JACOBIAN_SEPARATOR: ClassVar[str] = "!d$_$d!"
    """The string separating the input and output names in a derivative name.

    E.g. ``"output!d$_$d!input"``.
    """

    lock: RLock
    """The lock used for both multithreading and multiprocessing.

    Ensure safe multiprocessing and multithreading concurrent access to the cache.
    """

    lock_hashes: RLock
    """The lock used for both multithreading and multiprocessing.

    Ensure safe multiprocessing and multithreading concurrent access to the cache.
    """

    _hashes_to_indices: dict[int, ndarray]
    """The indices associated with the hashes."""

    _max_index: Value
    """The maximum index of the data stored in the cache."""

    _last_accessed_index: Value
    """The index of the last accessed data."""

    def __init__(  # noqa: D107
        self,
        tolerance: float = 0.0,
        name: str | None = None,
    ) -> None:
        super().__init__(tolerance, name)
        self.lock_hashes = RLock()
        self._hashes_to_indices = get_multi_processing_manager().dict()
        self._max_index = Value("i", 0)
        self._last_accessed_index = Value("i", 0)
        self.lock = self._set_lock()

    @abstractmethod
    def _set_lock(self) -> RLock:
        """Set a lock for multithreading.

        Either from an external object or internally by using RLock().
        """

    def __ensure_input_data_exists(
        self,
        input_data: Data,
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
        data_hash = hash_data_dict(input_data)

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
            if DATA_COMPARATOR(input_data, self._read_data(index, self._INPUTS_GROUP)):
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
        group: str,
    ) -> bool:
        """Check if an entry has data corresponding to a given group.

        Args:
            index: The index of the entry.
            group: The name of the group.

        Returns:
            Whether the entry has data for this group.
        """

    @abstractmethod
    def _write_data(
        self,
        values: Data,
        group: str,
        index: int,
    ) -> None:
        """Write the data associated with an index and a group.

        Args:
            values: The data containing the values of the names to cache.
            group: The name of the group,
                either :attr:`._INPUTS_GROUP`,
                :attr:`._OUTPUTS_GROUP`
                or :attr:`._JACOBIAN_GROUP`.
            index: The index of the entry in the cache.
        """

    def _cache_inputs(
        self,
        input_data: Data,
        group: str,
    ) -> bool:
        """Cache input data and increment group if needed.

        Cache inputs and increment group if needed.
        Check if ``group`` exists for these inputs.

        This method avoids duplicate storage.

        Args:
            input_data: The data containing the input data to cache.
            group: The name of the group to check the existence,
                either :attr:`._OUTPUTS_GROUP` or :attr:`._JACOBIAN_GROUP`.

        Returns:
            Whether ``group`` exists.
        """
        if self.__ensure_input_data_exists(input_data):
            self._write_data(input_data, self._INPUTS_GROUP, self._max_index.value)
        elif self._has_group(self._last_accessed_index.value, group):
            return True
        return False

    @synchronized
    def cache_outputs(  # noqa: D102
        self,
        input_data: Data,
        output_data: Data,
    ) -> None:
        if self._cache_inputs(input_data, self._OUTPUTS_GROUP):
            # There is already an output data corresponding to this input data.
            return

        self._write_data(
            output_data,
            self._OUTPUTS_GROUP,
            self._last_accessed_index.value,
        )

    @synchronized
    def cache_jacobian(  # noqa: D102
        self,
        input_data: Data,
        jacobian_data: JacobianData,
    ) -> None:
        if self._cache_inputs(input_data, self._JACOBIAN_GROUP):
            # There is already a Jacobian data corresponding to this input data.
            return

        flat_jacobian_data = flatten_nested_bilevel_dict(
            jacobian_data, separator=self._JACOBIAN_SEPARATOR
        )

        self._write_data(
            flat_jacobian_data,
            self._JACOBIAN_GROUP,
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
            self._read_data(self._last_accessed_index.value, self._INPUTS_GROUP),
            self._read_data(self._last_accessed_index.value, self._OUTPUTS_GROUP),
            self._read_data(self._last_accessed_index.value, self._JACOBIAN_GROUP),
        )

    @synchronized
    def __len__(self) -> int:
        return self._max_index.value

    @abstractmethod
    def _read_data(
        self,
        index: int,
        group: str,
        **options,
    ) -> Data | JacobianData:
        """Read the data of an entry.

        Args:
            index: The index of the entry.
            group: The name of the group to read.
            **options: The options passed to the overloaded methods.

        Returns:
            The output and Jacobian data corresponding to these index and group.
        """

    @synchronized_hashes
    def __has_hash(
        self,
        data_hash: int,
    ) -> ndarray | None:
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
        input_data: Data,
    ) -> CacheEntry:
        """Read the output and Jacobian data for a given input data.

        Args:
            indices: The indices of the entries among from which the entry to read data.
            input_data: The input data.

        Returns:
            The output and Jacobian data if they exist, ``None`` otherwise.
        """
        for index in indices:
            if DATA_COMPARATOR(input_data, self._read_data(index, self._INPUTS_GROUP)):
                output_data = self._read_data(index, self._OUTPUTS_GROUP)
                jacobian_data = self._read_data(index, self._JACOBIAN_GROUP)
                return CacheEntry(input_data, output_data, jacobian_data)

        return CacheEntry(input_data, {}, {})

    @synchronized
    def __getitem__(
        self,
        input_data: Data,
    ) -> CacheEntry:
        if self.tolerance == 0.0:
            data_hash = hash_data_dict(input_data)
            indices = self.__has_hash(data_hash)
            if indices is None:
                return CacheEntry(input_data, {}, {})

            return self._read_input_output_data(indices, input_data)

        for indices in self._hashes_to_indices.values():
            for index in indices:
                cached_input_data = self._read_data(index, self._INPUTS_GROUP)
                if DATA_COMPARATOR(input_data, cached_input_data, self.tolerance):
                    output_data = self._read_data(index, self._OUTPUTS_GROUP)
                    jacobian_data = self._read_data(index, self._JACOBIAN_GROUP)
                    return CacheEntry(input_data, output_data, jacobian_data)

        return CacheEntry(input_data, {}, {})

    @property
    def _all_groups(self) -> list[int]:
        """Sorted the indices of the entries."""
        return sorted(chain(*(v.tolist() for v in self._hashes_to_indices.values())))

    @synchronized
    def __iter__(self) -> Generator[CacheEntry]:
        return self._all_data()

    @synchronized
    def _all_data(self, **options) -> Generator[CacheEntry]:
        """Return an iterator of all data in the cache.

        Yields:
            The data position and the input, output and Jacobian data.
        """
        for index in self._all_groups:
            input_data = self._read_data(index, self._INPUTS_GROUP, **options)
            output_data = self._read_data(index, self._OUTPUTS_GROUP, **options)
            jacobian_data = self._read_data(index, self._JACOBIAN_GROUP, **options)
            yield CacheEntry(input_data, output_data, jacobian_data)

    def to_ggobi(
        self,
        file_path: str,
        input_names: Iterable[str] | None = None,
        output_names: Iterable[str] | None = None,
    ) -> None:
        """Export the cache to an XML file for ggobi tool.

        Args:
            file_path: The path of the file to export the cache.
            input_names: The names of the inputs to export.
                If ``None``, export all of them.
            output_names: The names of the outputs to export.
                If ``None``, export all of them.
        """
        if not self._hashes_to_indices:
            raise ValueError("An empty cache cannot be exported to XML file.")

        shared_input_names = None
        shared_output_names = None
        all_input_data = []
        all_output_data = []
        names_to_sizes = {}

        for data in self:
            input_data = data.inputs or {}
            output_data = data.outputs or {}
            try:
                if input_names is not None:
                    input_data = {name: input_data[name] for name in input_names}

                if output_names is not None:
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
            raise ValueError("Failed to find outputs in the cache.")

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
        other_cache: AbstractFullCache,
    ) -> None:
        """Update from another cache.

        Args:
            other_cache: The cache to update the current one.
        """
        for input_data, output_data, jacobian_data in other_cache:
            if output_data or jacobian_data:
                self[input_data] = (output_data, jacobian_data)

    @abstractmethod
    def _copy_empty_cache(self) -> AbstractFullCache:
        """Copy a cache without its entries."""

    def __add__(
        self,
        other_cache: AbstractFullCache,
    ) -> AbstractFullCache:
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


# TODO: API: remove dict from the method name.
def hash_data_dict(
    data: Mapping[str, ndarray | int | float],
) -> int:
    """Hash data using xxh3_64 from the xxhash library.

    Args:
        data: The data to hash.

    Returns:
        The hash value of the data.

    Examples:
        >>> from gemseo.core.cache import hash_data_dict
        >>> from numpy import array
        >>> data = {"x": array([1.0, 2.0]), "y": array([3.0])}
        >>> hash_data_dict(data)
        13252388834746642440
        >>> hash_data_dict(data, "x")
        4006190450215859422
    """
    names_with_hashed_values = []

    for name in sorted(data):
        value = data.get(name)
        if value is None:
            continue

        # xxh3_64 does not support int or float as input.
        if isinstance(value, ndarray):
            if value.dtype == int32 and sys.platform.startswith("win"):
                value = value.astype(int64)

            # xxh3_64 only supports C-contiguous arrays.
            if not value.flags["C_CONTIGUOUS"]:
                value = ascontiguousarray(value)
        else:
            value = array([value])

        value = value.view(uint8)

        hashed_value = xxh3_64_hexdigest(value)
        hashed_name = xxh3_64_hexdigest(bytes(name, "utf-8"))
        names_with_hashed_values.append((hashed_name, hashed_value))

    return int(xxh3_64_hexdigest(array(names_with_hashed_values)), 16)


def to_real(
    data: ndarray,
) -> ndarray:
    """Convert a NumPy array to a float NumPy array.

    Args:
        data: The NumPy array to be converted to real.

    Returns:
        A float NumPy array.
    """
    if data.dtype == complex128:
        return array(array(data, copy=False).real, dtype=float64)

    return data
