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
from abc import abstractmethod
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Mapping as ABCMapping
from collections.abc import Sized
from itertools import chain
from typing import TYPE_CHECKING
from typing import Literal
from typing import Protocol
from typing import overload

from numpy import hstack
from numpy import ndarray
from numpy import vstack
from pandas import MultiIndex
from strenum import StrEnum

from gemseo.caches.cache_entry import CacheEntry
from gemseo.datasets.dataset import Dataset
from gemseo.datasets.io_dataset import IODataset
from gemseo.typing import StrKeyMapping
from gemseo.utils.comparisons import compare_dict_of_arrays
from gemseo.utils.string_tools import MultiLineString

if TYPE_CHECKING:
    from gemseo.typing import JacobianData
    from gemseo.utils.comparisons import DataToCompare

    class DataComparator(Protocol):
        """A structural type for data comparator."""

        def __call__(  # noqa: D102
            self,
            dict_of_arrays: DataToCompare,
            other_dict_of_arrays: DataToCompare,
            tolerance: float = 0.0,
        ) -> bool: ...


LOGGER = logging.getLogger(__name__)


class BaseCache(ABCMapping[StrKeyMapping, CacheEntry]):
    """A base class for caches with a dictionary-like interface.

    Caches are mainly used to store the :class:`.Discipline` evaluations.

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

    _tolerance: float
    """The tolerance below which two input arrays are considered equal."""

    class Group(StrEnum):
        """A data group."""

        INPUTS = "inputs"
        """The label for the input variables."""

        OUTPUTS = "outputs"
        """The label for the output variables."""

        JACOBIAN = "jacobian"
        """The label for the Jacobian."""

    __names_to_sizes: dict[str, int]
    """The mapping from data names to sizes."""

    __input_names: list[str]
    """The names of the input data."""

    _output_names: list[str]
    """The names of the output data."""

    def __init__(
        self,
        tolerance: float = 0.0,
        name: str = "",
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
            name: A name for the cache. If empty, use the class name.
        """  # noqa: D205, D212, D415
        self._tolerance = tolerance
        self.name = name or self.__class__.__name__
        self.__names_to_sizes = {}
        self.__input_names = []
        self._output_names = []

    @property
    def tolerance(self) -> float:
        """The tolerance below which two input arrays are considered equal.

        Raises:
            ValueError: If the tolerance is not positive.
        """
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value: float) -> None:
        if value < 0.0:
            msg = f"The tolerance shall be positive: {value}"
            raise ValueError(msg)
        self._tolerance = value
        self._post_set_tolerance()

    @staticmethod
    def _post_set_tolerance() -> None:
        """Process after setting the tolerance, to be used by disciplines processes."""

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

    def _get_string_representation(self) -> MultiLineString:
        """The string representation of the cache."""
        mls = MultiLineString()
        mls.add("Name: {}", self.name)
        mls.indent()
        mls.add("Type: {}", self.__class__.__name__)
        mls.add("Tolerance: {}", self._tolerance)
        mls.add("Input names: {}", self.input_names)
        mls.add("Output names: {}", self.output_names)
        mls.add("Length: {}", len(self))
        return mls

    def __repr__(self) -> str:
        return str(self._get_string_representation())

    def _repr_html_(self) -> str:
        return self._get_string_representation()._repr_html_()

    def __setitem__(
        self,
        input_data: StrKeyMapping,
        data: tuple[StrKeyMapping, JacobianData],
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
    def cache_outputs(
        self,
        input_data: StrKeyMapping,
        output_data: StrKeyMapping,
    ) -> None:
        """Cache input and output data.

        Args:
            input_data: The data containing the input data to cache.
            output_data: The data containing the output data to cache.
        """

    @abstractmethod
    def cache_jacobian(
        self,
        input_data: StrKeyMapping,
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

    @overload
    def to_dataset(
        self,
        name: str = ...,
        categorize: Literal[True] = ...,
        input_names: Iterable[str] = ...,
        output_names: Iterable[str] = ...,
    ) -> IODataset: ...

    @overload
    def to_dataset(
        self,
        name: str = ...,
        categorize: Literal[False] = ...,
        input_names: Iterable[str] = ...,
        output_names: Iterable[str] = ...,
    ) -> Dataset: ...

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
        dataset_class: type[Dataset]

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
                for cache_entry in self.get_all_entries():
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

    @abstractmethod
    def get_all_entries(self) -> Iterator[CacheEntry]:
        """Return an iterator over all the entries.

        The tolerance is ignored.

        Yields:
            The entries.
        """

    # TODO: API: make it behave like mappings, ie. like .keys().
    def __iter__(self) -> Iterator[CacheEntry]:  # type: ignore[override]
        return self.get_all_entries()

    compare_dict_of_arrays = staticmethod(compare_dict_of_arrays)
