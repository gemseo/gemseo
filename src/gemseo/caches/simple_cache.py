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

from gemseo.core.cache import AbstractCache
from gemseo.core.cache import CacheEntry
from gemseo.core.cache import Data
from gemseo.core.cache import JacobianData
from gemseo.utils.data_conversion import deepcopy_dict_of_arrays
from gemseo.utils.testing import compare_dict_of_arrays


class SimpleCache(AbstractCache):
    """Dictionary-based cache storing a unique entry.

    When caching an input data different from this entry, this entry is replaced by a new
    one initialized with this input data.
    """

    def __init__(  # noqa:D107
        self,
        tolerance: float = 0.0,
        name: str | None = None,
    ) -> None:
        super().__init__(tolerance, name)
        self.__input_data_for_outputs = {}
        self.__output_data = {}
        self.__input_data_for_jacobian = {}
        self.__jacobian_data = {}
        self.__last_input_data = {}
        self.__penultimate_input_data = {}

    def clear(self) -> None:  # noqa:D102
        super().clear()
        self.__input_data_for_outputs = {}
        self.__output_data = {}
        self.__input_data_for_jacobian = {}
        self.__jacobian_data = {}
        self.__last_input_data = {}
        self.__penultimate_input_data = {}

    def __iter__(self) -> Generator[CacheEntry]:
        if self.penultimate_entry.inputs:
            yield self.penultimate_entry

        yield self.last_entry

    def __len__(self) -> int:
        return bool(self.__penultimate_input_data) + bool(self.__last_input_data)

    def __create_input_cache(self, input_data: Data) -> Data:
        """Create the input data.

        Args:
            input_data: The data containing the input data to cache.

        Returns:
            A copy of the input data.
        """
        cached_input_data = deepcopy_dict_of_arrays(input_data)
        if not self.__is_cached(self.__last_input_data, cached_input_data):
            self.__penultimate_input_data = self.__last_input_data
            self.__last_input_data = cached_input_data

        return cached_input_data

    def __is_cached(
        self,
        cached_input_data: Data,
        input_data: Data,
    ) -> bool:
        """Check if an input data is cached.

        Args:
            cached_input_data: The cached input data.
            input_data: The input data to be verified.

        Returns:
            Whether the input data is cached.
        """
        if not cached_input_data:
            return False

        if self.tolerance == 0.0:
            if compare_dict_of_arrays(input_data, cached_input_data):
                return True

        else:
            if compare_dict_of_arrays(input_data, cached_input_data, self.tolerance):
                return True

        return False

    def cache_outputs(  # noqa:D102
        self,
        input_data: Data,
        output_data: Data,
    ) -> None:
        self.__input_data_for_outputs = self.__create_input_cache(input_data)
        self.__output_data = deepcopy_dict_of_arrays(output_data)
        if not self._output_names:
            self._output_names = sorted(output_data.keys())

    def __getitem__(
        self,
        input_data: Data,
    ) -> CacheEntry:
        output_data, jacobian_data = {}, {}
        if self.__is_cached(self.__input_data_for_outputs, input_data):
            output_data = self.__output_data

        if self.__is_cached(self.__input_data_for_jacobian, input_data):
            jacobian_data = self.__jacobian_data

        return CacheEntry(input_data, output_data, jacobian_data)

    def cache_jacobian(  # noqa:D102
        self,
        input_data: Data,
        jacobian_data: JacobianData,
    ) -> None:
        self.__input_data_for_jacobian = self.__create_input_cache(input_data)
        self.__jacobian_data = jacobian_data

    def __retrieve_entry(
        self,
        cached_input_data: Data,
    ) -> CacheEntry:
        """Return the cache entry corresponding to a cached input data.

        Args:
            cached_input_data: The cached input data.

        Returns:
            The cache entry corresponding to this cached input data.
        """
        input_data = cached_input_data
        output_data = {}
        jacobian_data = {}
        if self.__is_cached(self.__input_data_for_outputs, cached_input_data):
            input_data = self.__input_data_for_outputs
            output_data = self.__output_data

        if self.__is_cached(self.__input_data_for_jacobian, cached_input_data):
            input_data = self.__input_data_for_jacobian
            jacobian_data = self.__jacobian_data

        return CacheEntry(input_data, output_data, jacobian_data)

    @property
    def penultimate_entry(self) -> CacheEntry:
        """The penultimate cache entry."""
        entry = self.__retrieve_entry(self.__penultimate_input_data)
        if entry.outputs or entry.jacobian:
            return entry

        return CacheEntry({}, {}, {})

    @property
    def last_entry(self) -> CacheEntry:  # noqa:D102
        return self.__retrieve_entry(self.__last_input_data)
