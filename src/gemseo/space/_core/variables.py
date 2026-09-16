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
"""Versioned variables."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import TYPE_CHECKING
from typing import TypeVar

from numpy import concatenate
from numpy import full
from numpy import zeros

from gemseo.space.variable import BaseVariable
from gemseo.space.variable import DataType
from gemseo.util.metaclass import ABCGoogleDocstringInheritanceMeta
from gemseo.util.read_only_mapping import ReadOnlyMapping

if TYPE_CHECKING:
    from collections.abc import Iterator
    from collections.abc import Sequence
    from typing import Any

    from gemseo.util.typing import BooleanArray


_VariableT = TypeVar("_VariableT", bound=BaseVariable)
"""The type of the variables of the registry."""


class UnknownVariableError(KeyError):
    """Raised when accessing a variable name absent from the registry."""

    def __str__(self) -> str:
        return self.args[0]


def check_components(name: str, components: Sequence[int]) -> None:
    """Check that at least one component of a variable is kept.

    This is shared by the registries so that every kind of space reports
    an empty selection of components with the same message.

    Args:
        name: The name of the variable.
        components: The components to be kept.

    Raises:
        ValueError: When no component is to be kept.
    """
    if not components:
        msg = f"A variable cannot be empty; got no component for {name!r}."
        raise ValueError(msg)


class Variables(
    MutableMapping[str, _VariableT], metaclass=ABCGoogleDocstringInheritanceMeta
):
    """A registry of [BaseVariable][gemseo.space.variable.BaseVariable] objects.

    This registry is ordered and versioned.

    It is the single source of truth for
    *which* variables exist in a variable space,
    their order
    and their sizes.

    Every mutation increments `version`
    so downstream consumers that derive values from it
    (bounds arrays, normalization factors, integer masks)
    can detect staleness by comparing against this monotonic integer.

    The values of the variables are treated as vectors.
    Their concatenation is called the full vector.
    Its value is referred to as the full value.

    The mappings whose keys are variable names are sorted
    in the order in which the variables were added.

    The registry is itself a
    [MutableMapping][collections.abc.MutableMapping] from a variable name to a
    [BaseVariable][gemseo.space.variable.BaseVariable]:
    read it with `registry[name]`, `.keys()`, `.values()`, `.items()`, `.get()`,
    iteration, membership and length;
    insert or replace a variable with `registry[name] = variable`
    (a new name is appended, an existing one keeps its position;
    the index ranges and `size` are rebuilt and `version` is bumped);
    delete a variable with `del registry[name]`.
    Every write bumps `version` so downstream consumers can detect staleness.

    The operations that do not map onto item assignment or deletion,
    namely `rename` and `filter_components`,
    remain explicit methods.

    Note:
        This registry is internal to a space of variables,
        which owns the data derived from it, e.g. the current value of a design space,
        and so is the only writer.
        A space exposes it to its users
        as a [VariablesView][gemseo.space.variables_view.VariablesView],
        which is read-only.
    """

    __name_to_variable: dict[str, _VariableT]
    """The map from a variable name to a variable."""

    __name_to_indices: dict[str, range]
    """The map from a variable name to an index range in the full vector."""

    __size: int
    """The size of the full vector."""

    __version: int
    """The version number of the variables."""

    name_to_indices: ReadOnlyMapping[str, range]
    """The map from a variable name to an index range in the full vector (read-only)."""

    def __init__(self) -> None:  # noqa: D107
        self.__name_to_variable = {}
        self.__name_to_indices = {}
        self.__size = 0
        self.__version = 0
        self.name_to_indices = ReadOnlyMapping(self.__name_to_indices)

    @property
    def size(self) -> int:
        """The size of the full vector."""
        return self.__size

    @property
    def version(self) -> int:
        """The version number of the variables."""
        return self.__version

    def bump_version(self) -> None:
        """Increment the version number."""
        self.__version += 1

    def __setitem__(self, name: str, variable: _VariableT) -> None:
        # Insert a new variable (appended) or replace an existing one (in place),
        # possibly changing its size; the index ranges and size are rebuilt.
        self.__name_to_variable[name] = variable
        self.__reindex()
        self.bump_version()

    def __delitem__(self, name: str) -> None:
        # Validate via __getitem__ so an unknown name raises with a clear message.
        self[name]
        del self.__name_to_variable[name]
        del self.__name_to_indices[name]
        self.__reindex()
        self.bump_version()

    def __reindex(self) -> None:
        """Rebuild the contiguous index ranges and the full-vector size from scratch."""
        start = 0
        for name, variable in self.__name_to_variable.items():
            self.__name_to_indices[name] = range(start, start + variable.size)
            start += variable.size
        self.__size = start

    def rename(self, current_name: str, new_name: str) -> None:
        """Rename a variable.

        Args:
            current_name: The name of the variable to rename.
            new_name: The new name of the variable.

        Raises:
            ValueError: When `new_name` is already the name of another variable.
        """
        # Validate via __getitem__ so an unknown name raises before mutating.
        self[current_name]
        if new_name != current_name and new_name in self:
            msg = (
                f"Cannot rename {current_name!r} to {new_name!r}: "
                f"{new_name!r} is already the name of another variable."
            )
            raise ValueError(msg)

        self._rename_key(self.__name_to_variable, current_name, new_name)
        self._rename_key(self.__name_to_indices, current_name, new_name)
        self.bump_version()

    @staticmethod
    def _rename_key(mapping: dict[str, Any], current_name: str, new_name: str) -> None:
        """Rename a key of a mapping in place, preserving its object identity.

        Args:
            mapping: The mapping.
            current_name: The key to rename.
            new_name: The new key.
        """
        items = [
            (new_name if name == current_name else name, value)
            for name, value in mapping.items()
        ]
        mapping.clear()
        mapping.update(items)

    def filter_components(self, name: str, components: Sequence[int]) -> None:
        """Keep only certain components of a variable.

        Args:
            name: The name of the variable.
            components: The components to be kept.

        Raises:
            ValueError: When no component is to be kept.

        Note:
            This method increments the version number.
        """
        variable = self[name]
        check_components(name, components)
        # The variable knows how to restrict itself to some of its components,
        # so that a kind whose bounds are derived is not rebuilt
        # from bounds passed explicitly.
        self.__name_to_variable[name] = variable.filter_components(components)
        self.__reindex()
        self.bump_version()

    def get_integer_mask(self) -> BooleanArray:
        """Return whether the components of the full vector are integer.

        Returns:
            Whether the components of the full vector are integer
            (one result per component).
        """
        if not self:
            return zeros(0, dtype=bool)

        return concatenate(
            tuple(
                full(variable.size, variable.type == DataType.INTEGER, dtype=bool)
                for variable in self.values()
            )
        )

    @property
    def has_integer_variables(self) -> bool:
        """Whether the registry has at least one integer variable."""
        return any(variable.type == DataType.INTEGER for variable in self.values())

    @property
    def has_discrete_variables(self) -> bool:
        """Whether the registry has at least one discrete variable."""
        return any(variable.type == DataType.DISCRETE for variable in self.values())

    def __getitem__(self, name: str) -> _VariableT:
        try:
            return self.__name_to_variable[name]
        except KeyError:
            msg = f"No variable named {name!r}."
            raise UnknownVariableError(msg) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self.__name_to_variable)

    def __len__(self) -> int:
        return len(self.__name_to_variable)
