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
"""Integer-component rounding for versioned variables."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import round as np_round

from gemseo.space.design._registry_derived_data import RegistryDerivedData

if TYPE_CHECKING:
    from numpy import ndarray

    from gemseo.space.design._variables import Variables
    from gemseo.util.typing import BooleanArray


class IntegerRounder(RegistryDerivedData):
    """Rounding of the integer-typed components of versioned variables."""

    __integer_mask: BooleanArray | None
    """The integer-mask of the full vector."""

    __no_integer: bool
    """Whether the full vector has no integer component."""

    def __init__(self, variables: Variables) -> None:
        """
        Args:
            variables: The variables.
        """  # noqa: D205, D212
        super().__init__(variables)
        self._register_guard(self._rebuild)
        self.__integer_mask = None
        self.__no_integer = True

    def _rebuild(self) -> None:
        """Rebuild the integer-mask of the full vector."""
        self.__integer_mask = self._variables.get_integer_mask()
        self.__no_integer = (
            not self.__integer_mask.any() if self.__integer_mask.size else True
        )

    @property
    def has_integer(self) -> bool:
        """Whether the full vector has at least one integer component."""
        self._refresh()
        return not self.__no_integer

    def round(self, full_value: ndarray, copy: bool = True) -> ndarray:
        """Round the integer components of a full value.

        Args:
            full_value: The full value.
            copy: Whether to round a copy of `full value`.

        Returns:
            The rounded values.
        """
        self._refresh()
        if self.__no_integer:
            return full_value

        integer_mask = self.__integer_mask
        rounded_value = full_value.copy() if copy else full_value
        rounded_value[..., integer_mask] = np_round(full_value[..., integer_mask])
        return rounded_value
