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
"""Versioned design variables."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.space._core.variables import Variables
from gemseo.space.variable import BaseDeterministicVariable
from gemseo.util._numpy import freeze_array
from gemseo.util.read_only_mapping import ReadOnlyMapping
from gemseo.util.typing import BooleanArray

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any


class NormalizationMasks(ReadOnlyMapping[str, BooleanArray]):
    """A read-only live view over the normalization masks of a registry.

    The masks are frozen,
    but NumPy lets a caller reassign the shape, the strides and the data type
    of a read-only array, and these belong to the array object itself,
    so this view hands out a view of a mask rather than the mask stored;
    a reassignment then reaches the view of the caller only.
    """

    __slots__ = ()

    def __getitem__(self, key: str) -> BooleanArray:
        return self._mapping[key].view()


# The design variables are deterministic, and today they all happen to be
# numeric too, so the helpers of this package read their bounds directly;
# a kind whose components are not numbers, e.g. a categorical one, will make
# these reads conditional on isinstance(variable, BaseNumericVariable).
class DesignVariables(Variables[BaseDeterministicVariable]):
    """A registry of design variables.

    In addition to the generic registry behavior,
    this registry is the single source of truth for
    the per-component normalization policy of the design variables.
    """

    __name_to_normalization_mask: dict[str, BooleanArray]
    """The map from a variable name to a per-component normalization policy mask."""

    __enable_integer_variables_normalization: bool
    """Whether to normalize integer variables."""

    name_to_normalization_mask: ReadOnlyMapping[str, BooleanArray]
    """The map from a variable name to a per-component normalization policy mask (read-only)."""  # noqa: E501

    def __init__(self) -> None:  # noqa: D107
        self.__name_to_normalization_mask = {}
        self.__enable_integer_variables_normalization = False
        self.name_to_normalization_mask = NormalizationMasks(
            self.__name_to_normalization_mask
        )
        super().__init__()

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the registry from a pickled or copied state.

        Args:
            state: The state.
        """
        self.__dict__.update(state)
        # Pickling preserves neither the writeable flag nor the immutable buffer
        # that a frozen array is a view of, so freeze the masks again.
        for name, mask in self.__name_to_normalization_mask.items():
            self.__name_to_normalization_mask[name] = freeze_array(mask)

    @property
    def enable_integer_variables_normalization(self) -> bool:
        """Whether to normalize the integer variables.

        Note:
            Setting this attribute increments the version number.
        """
        return self.__enable_integer_variables_normalization

    @enable_integer_variables_normalization.setter
    def enable_integer_variables_normalization(self, value: bool) -> None:
        if value == self.__enable_integer_variables_normalization:
            return

        self.__enable_integer_variables_normalization = value
        # The policy of a variable whose kind ignores the flag is unchanged,
        # so recomputing it for every variable is behavior-preserving.
        for name, variable in self.items():
            self.__name_to_normalization_mask[name] = self.__get_normalization_mask(
                variable
            )

        self.bump_version()

    def __setitem__(self, name: str, variable: BaseDeterministicVariable) -> None:
        super().__setitem__(name, variable)
        self.__name_to_normalization_mask[name] = self.__get_normalization_mask(
            variable
        )

    def __delitem__(self, name: str) -> None:
        super().__delitem__(name)
        del self.__name_to_normalization_mask[name]

    def rename(self, current_name: str, new_name: str) -> None:  # noqa: D102
        super().rename(current_name, new_name)
        self._rename_key(self.__name_to_normalization_mask, current_name, new_name)

    def filter_components(self, name: str, components: Sequence[int]) -> None:  # noqa: D102
        super().filter_components(name, components)
        self.__name_to_normalization_mask[name] = self.__get_normalization_mask(
            self[name]
        )

    def __get_normalization_mask(
        self, variable: BaseDeterministicVariable
    ) -> BooleanArray:
        """Return the normalization policy mask of a variable.

        The policy belongs to the kind of the variable;
        this method only forwards the integer-normalization setting of the set to
        [BaseDeterministicVariable.get_normalization_mask][gemseo.space.variable.deterministic.BaseDeterministicVariable.get_normalization_mask].

        Args:
            variable: The variable.

        Returns:
            The per-component normalization mask (read-only).
        """
        mask = variable.get_normalization_mask(
            self.__enable_integer_variables_normalization
        )

        # Freeze a writeable mask, in case a kind of variable hands one out,
        # so that mutating a mask cannot bypass the version bump
        # and leave the derived caches serving stale normalization policies;
        # a frozen mask is stored as it is.
        return freeze_array(mask)
