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
"""Caching module to store all the entries in memory."""

from __future__ import annotations

from copy import copy
from multiprocessing import RLock
from typing import TYPE_CHECKING
from typing import Literal
from typing import cast
from typing import overload

from gemseo.caches.base_full_cache import BaseFullCache
from gemseo.utils.data_conversion import nest_flat_bilevel_dict
from gemseo.utils.locks import synchronized
from gemseo.utils.multiprocessing.manager import get_multi_processing_manager

if TYPE_CHECKING:
    from multiprocessing.managers import DictProxy
    from multiprocessing.synchronize import RLock as RLockType

    from gemseo.typing import JacobianData
    from gemseo.typing import StrKeyMapping


class MemoryFullCache(BaseFullCache):
    """Cache using memory to cache all the data."""

    __data: (
        DictProxy[int, dict[BaseFullCache.Group, StrKeyMapping | JacobianData]]
        | dict[int, dict[BaseFullCache.Group, StrKeyMapping | JacobianData]]
    )

    def __init__(
        self,
        tolerance: float = 0.0,
        name: str = "",
        is_memory_shared: bool = True,
    ) -> None:
        """
        Args:
            is_memory_shared : If ``True``,
                a shared memory dictionary is used to store the data,
                which makes the cache compatible with multiprocessing.

        Warnings:
            If ``is_memory_shared`` is ``False``
            and multiple disciplines point to the same cache
            or the process is multi-processed,
            there may be duplicate computations
            because the cache will not be shared among the processes.
            This class relies on some multiprocessing features, it is therefore
            necessary to protect its execution with an ``if __name__ == '__main__':``
            statement when working on Windows.
        """  # noqa: D205, D212, D415
        super().__init__(tolerance, name)
        self.__is_memory_shared = is_memory_shared
        if self.__is_memory_shared:
            self.__data = get_multi_processing_manager().dict()
        else:
            self.__data = {}

    def _copy_empty_cache(self) -> MemoryFullCache:
        return MemoryFullCache(self._tolerance, self.name, self.__is_memory_shared)

    def _initialize_entry(
        self,
        index: int,
    ) -> None:
        self.__data[index] = {}

    def _set_lock(self) -> RLockType:
        return RLock()

    def _has_group(
        self,
        index: int,
        group: BaseFullCache.Group,
    ) -> bool:
        return group in self.__data[index]

    @synchronized
    def clear(self) -> None:  # noqa:D102
        super().clear()
        self.__data.clear()

    @overload
    def _read_data(
        self,
        index: int,
        group: Literal[BaseFullCache.Group.INPUTS, BaseFullCache.Group.OUTPUTS],
    ) -> StrKeyMapping: ...

    @overload
    def _read_data(
        self,
        index: int,
        group: Literal[BaseFullCache.Group.JACOBIAN],
    ) -> JacobianData: ...

    def _read_data(
        self,
        index: int,
        group: BaseFullCache.Group,
    ) -> StrKeyMapping | JacobianData:
        data = self.__data[index].get(group, {})
        if group == self.Group.JACOBIAN and data:
            return nest_flat_bilevel_dict(
                cast("JacobianData", data), separator=self._JACOBIAN_SEPARATOR
            )
        return data

    def _write_data(
        self,
        values: StrKeyMapping,
        group: BaseFullCache.Group,
        index: int,
    ) -> None:
        data = self.__data[index]
        data[group] = copy(values)
        self.__data[index] = data

    @property
    def copy(self) -> MemoryFullCache:
        """Copy the current cache.

        Returns:
            A copy of the current cache.
        """
        cache = self._copy_empty_cache()
        cache.update(self)
        return cache
