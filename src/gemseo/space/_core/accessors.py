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
"""Shortcuts to the variables of a space."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Generic
from typing import TypeVar

from gemseo.space._core.variables import Variables

if TYPE_CHECKING:
    from gemseo.util.read_only_mapping import ReadOnlyMapping

_VariablesT = TypeVar("_VariablesT", bound=Variables)
"""The type of the registry of the variables of the space."""


# TODO: API: remove this mixin and rewrite its call sites against the
# variables view, which every space has and which this mixin only duplicates:
# variable_names -> list(space.variables),
# variable_sizes -> {n: v.size for n, v in space.variables.items()},
# variable_types -> the same with v.type,
# name_to_indices -> space.variables.name_to_indices,
# get_size(n) -> space.variables[n].size,
# get_type(n) -> str(space.variables[n].type),
# has_integer_variables -> any(
#     v.type == DataType.INTEGER for v in space.variables.values()
# ).
# A RandomSpace deliberately does not use this mixin, so a consumer typed
# BaseVariableSpace can only use the view; keeping the mixin makes the two
# spaces expose different accessors for the same data.
class VariableAccessorsMixin(Generic[_VariablesT]):
    """Shortcuts to the names, sizes, types and indices of the variables of a space.

    These accessors are conveniences over the
    [variables][gemseo.space.base.BaseVariableSpace.variables] view.
    Reading a variable through the view does not build a new list or dictionary,
    so prefer it in a loop.

    A class using this mixin must be a
    [BaseVariableSpace][gemseo.space.base.BaseVariableSpace].
    """

    __slots__ = ()

    _variables: _VariablesT
    """The versioned variables."""

    @property
    def variable_names(self) -> list[str]:
        """The variable names."""
        return list(self._variables)

    @property
    def variable_sizes(self) -> dict[str, int]:
        """The variable sizes."""
        return {name: variable.size for name, variable in self._variables.items()}

    @property
    def variable_types(self) -> dict[str, str]:
        """The variable types."""
        return {name: variable.type for name, variable in self._variables.items()}

    @property
    def name_to_indices(self) -> ReadOnlyMapping[str, range]:
        """The names bound to the indices."""
        return self._variables.name_to_indices

    @property
    def has_integer_variables(self) -> bool:
        """Whether the space has at least one integer variable."""
        return self._variables.has_integer_variables

    def get_size(self, name: str) -> int:
        """Get the size of a variable.

        Args:
            name: The name of the variable.

        Returns:
            The size of the variable.
        """
        return self._variables[name].size

    def get_type(self, name: str) -> str:
        """Return the type of a variable.

        Args:
            name: The name of the variable.

        Returns:
            The type of the variable.
        """
        return str(self._variables[name].type)
