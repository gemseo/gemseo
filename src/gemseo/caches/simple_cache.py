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

from typing import TYPE_CHECKING

from gemseo.caches.base_cache import BaseCache
from gemseo.caches.cache_entry import CacheEntry
from gemseo.utils.data_conversion import deepcopy_dict_of_arrays

if TYPE_CHECKING:
    from collections.abc import Iterator

    from gemseo.typing import JacobianData
    from gemseo.typing import StrKeyMapping


class SimpleCache(BaseCache):
    """Dictionary-based cache storing a unique entry."""

    __inputs: StrKeyMapping
    """The input data."""

    __outputs: StrKeyMapping
    """The output data."""

    __jacobian: JacobianData
    """The Jacobian data."""

    def __init__(  # noqa:D107
        self,
        tolerance: float = 0.0,
        name: str = "",
    ) -> None:
        super().__init__(tolerance, name)
        self.clear()

    def clear(self) -> None:  # noqa:D102
        super().clear()
        self.__inputs = {}
        self.__outputs = {}
        self.__jacobian = {}

    def get_all_entries(self) -> Iterator[CacheEntry]:  # noqa: D102
        if self.__inputs:
            yield self.last_entry

    def __len__(self) -> int:
        return 1 if self.__inputs else 0

    def __is_cached(
        self,
        input_data: StrKeyMapping,
    ) -> bool:
        """Check if an input data is cached.

        Args:
            input_data: The input data to be verified.

        Returns:
            Whether the input data is cached.
        """
        return len(self.__inputs) != 0 and self.compare_dict_of_arrays(
            input_data, self.__inputs, self._tolerance
        )

    def cache_outputs(  # noqa:D102
        self,
        input_data: StrKeyMapping,
        output_data: StrKeyMapping,
    ) -> None:
        if self.__is_cached(input_data):
            if not self.__outputs:
                self.__outputs = deepcopy_dict_of_arrays(output_data)
            return

        self.__inputs = deepcopy_dict_of_arrays(input_data)
        self.__outputs = deepcopy_dict_of_arrays(output_data)
        self.__jacobian = {}

        if not self._output_names:
            self._output_names = sorted(output_data.keys())

    def __getitem__(
        self,
        input_data: StrKeyMapping,
    ) -> CacheEntry:
        if not self.__is_cached(input_data):
            return CacheEntry(input_data, {}, {})
        return self.last_entry

    def cache_jacobian(  # noqa:D102
        self,
        input_data: StrKeyMapping,
        jacobian_data: JacobianData,
    ) -> None:
        if self.__is_cached(input_data):
            if not self.__jacobian:
                self.__jacobian = jacobian_data
            return

        self.__inputs = deepcopy_dict_of_arrays(input_data)
        self.__jacobian = jacobian_data
        self.__outputs = {}

    @property
    def last_entry(self) -> CacheEntry:  # noqa:D102
        return CacheEntry(self.__inputs, self.__outputs, self.__jacobian)
