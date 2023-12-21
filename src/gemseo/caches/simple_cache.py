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

from gemseo.core.cache import DATA_COMPARATOR
from gemseo.core.cache import AbstractCache
from gemseo.core.cache import CacheEntry
from gemseo.core.cache import JacobianData
from gemseo.utils.data_conversion import deepcopy_dict_of_arrays

if TYPE_CHECKING:
    from collections.abc import Generator
    from collections.abc import Mapping

    from numpy import ndarray

    from gemseo.core.discipline_data import Data


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
        return self.__inputs and DATA_COMPARATOR(
            input_data, self.__inputs, self.tolerance
        )

    def cache_outputs(  # noqa:D102
        self,
        input_data: Data,
        output_data: Data,
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
