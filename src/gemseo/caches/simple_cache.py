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
"""Caching module to store only one entry."""
from __future__ import annotations

from typing import Generator
from typing import Mapping

from numpy import ndarray

from gemseo.core.cache import AbstractCache
from gemseo.core.cache import CacheEntry
from gemseo.core.cache import Data
from gemseo.core.cache import JacobianData
from gemseo.utils.comparisons import compare_dict_of_arrays
from gemseo.utils.data_conversion import deepcopy_dict_of_arrays


class SimpleCache(AbstractCache):
    """Dictionary-based cache storing a unique entry."""

    __inputs: Mapping[str, ndarray]
    """The input data."""

    __outputs: Mapping[str, ndarray]
    """The output data."""

    __jacobian: Mapping[str, Mapping[str, ndarray]]
    """The Jacobian data."""

    def __init__(  # noqa:D107
        self,
        tolerance: float = 0.0,
        name: str | None = None,
    ) -> None:
        super().__init__(tolerance, name)
        self.clear()

    def clear(self) -> None:  # noqa:D102
        super().clear()
        self.__inputs = {}
        self.__outputs = {}
        self.__jacobian = {}

    def __iter__(self) -> Generator[CacheEntry]:
        if self.__inputs:
            yield self.last_entry

    def __len__(self) -> int:
        return 1 if self.__inputs else 0

    def __cache_inputs(self, input_data: Data) -> None:
        """Cache the input data.

        Args:
            input_data: The input data to cache.
        """
        cached_input_data = deepcopy_dict_of_arrays(input_data)
        if not self.__is_cached(cached_input_data):
            self.__inputs = cached_input_data
            self.__outputs = {}
            self.__jacobian = {}

    def __is_cached(
        self,
        input_data: Data,
    ) -> bool:
        """Check if an input data is cached.

        Args:
            input_data: The input data to be verified.

        Returns:
            Whether the input data is cached.
        """
        cached_input_data = self.__inputs
        if not cached_input_data:
            return False
        return compare_dict_of_arrays(input_data, cached_input_data, self.tolerance)

    def cache_outputs(  # noqa:D102
        self,
        input_data: Data,
        output_data: Data,
    ) -> None:
        self.__cache_inputs(input_data)
        self.__outputs = deepcopy_dict_of_arrays(output_data)
        if not self._output_names:
            self._output_names = sorted(output_data.keys())

    def __getitem__(
        self,
        input_data: Data,
    ) -> CacheEntry:
        if not self.__is_cached(input_data):
            return CacheEntry(input_data, {}, {})
        return self.last_entry

    def cache_jacobian(  # noqa:D102
        self,
        input_data: Data,
        jacobian_data: JacobianData,
    ) -> None:
        self.__cache_inputs(input_data)
        self.__jacobian = jacobian_data

    @property
    def last_entry(self) -> CacheEntry:  # noqa:D102
        return CacheEntry(self.__inputs, self.__outputs, self.__jacobian)
