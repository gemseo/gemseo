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

from numpy import concatenate
from numpy import full
from numpy import inf
from numpy import logical_and
from numpy import zeros

from gemseo.space._variable import DataType
from gemseo.space._variable import Variable
from gemseo.util.metaclass import ABCGoogleDocstringInheritanceMeta
from gemseo.util.read_only_mapping import ReadOnlyMapping

if TYPE_CHECKING:
    from collections.abc import Iterator
    from collections.abc import Sequence
    from typing import Any

    from gemseo.util.typing import BooleanArray


class UnknownVariableError(KeyError):
    """Raised when accessing a variable name absent from the registry."""

    def __str__(self) -> str:
        return self.args[0]


class Variables(
    MutableMapping[str, Variable], metaclass=ABCGoogleDocstringInheritanceMeta
):
    """A registry of [Variable][gemseo.space._variable.Variable] objects.

    This registry is ordered and versioned.

    It is the single source of truth for
    *which* variables exist in a [DesignSpace][gemseo.space.design.DesignSpace],
    their order,
    their sizes,
    and their per-component normalization policy.

    Every mutation increments `version`
    so downstream consumers that derive values from it
    (bounds arrays, normalization factors, integer masks)
    can detect staleness by comparing against this monotonic integer.

    The values of the variables are treated as vectors.
    Their concatenation is called the full vector.
    Its value is referred to as full value.

    The mappings whose keys are variable names are sorted
    in the order in which the variables were added.

    The registry is itself a
    [MutableMapping][collections.abc.MutableMapping] from a variable name to a
    [Variable][gemseo.space._variable.Variable]:
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
    """

    __name_to_variable: dict[str, Variable]
    """The map from a variable name to a variable."""

    __name_to_indices: dict[str, range]
    """The map from a variable name to an index range in the full vector."""

    __name_to_normalization_mask: dict[str, BooleanArray]
    """The map from a variable name to a per-component normalization policy mask."""

    __size: int
    """The size of the full vector."""

    __version: int
    """The version number of the variables."""

    __enable_integer_variables_normalization: bool
    """Whether to normalize integer variables."""

    name_to_indices: ReadOnlyMapping[str, range]
    """The map from a variable name to an index range in the full vector (read-only)."""

    name_to_normalization_mask: ReadOnlyMapping[str, BooleanArray]
    """The map from a variable name to a per-component normalization policy mask (read-only)."""  # noqa: E501

    def __init__(self) -> None:  # noqa: D107
        self.__name_to_variable = {}
        self.__name_to_indices = {}
        self.__name_to_normalization_mask = {}
        self.__size = 0
        self.__version = 0
        self.__enable_integer_variables_normalization = False
        self.name_to_indices = ReadOnlyMapping(self.__name_to_indices)
        self.name_to_normalization_mask = ReadOnlyMapping(
            self.__name_to_normalization_mask
        )

    @property
    def size(self) -> int:
        """The size of the full vector."""
        return self.__size

    @property
    def version(self) -> int:
        """The version number of the variables."""
        return self.__version

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
        for name, variable in self.__name_to_variable.items():
            if variable.type == DataType.INTEGER:
                self.__name_to_normalization_mask[name] = (
                    self.__compute_normalization_mask(variable)
                )

        self.bump_version()

    def bump_version(self) -> None:
        """Increment the version number."""
        self.__version += 1

    def __setitem__(self, name: str, variable: Variable) -> None:
        # Insert a new variable (appended) or replace an existing one (in place),
        # possibly changing its size; the index ranges and size are rebuilt.
        self.__name_to_variable[name] = variable
        self.__name_to_normalization_mask[name] = self.__compute_normalization_mask(
            variable
        )
        self.__reindex()
        self.bump_version()

    def __delitem__(self, name: str) -> None:
        # Validate via __getitem__ so an unknown name raises with a clear message.
        self[name]
        del self.__name_to_variable[name]
        del self.__name_to_normalization_mask[name]
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
        """
        # Validate via __getitem__ so an unknown name raises before mutating.
        self[current_name]
        self.__rename_key(self.__name_to_normalization_mask, current_name, new_name)
        self.__rename_key(self.__name_to_variable, current_name, new_name)
        self.__rename_key(self.__name_to_indices, current_name, new_name)
        self.bump_version()

    @staticmethod
    def __rename_key(mapping: dict[str, Any], current_name: str, new_name: str) -> None:
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

        Note:
            This method increments the version number.
        """
        variable = self[name]
        idx = list(components)
        new_variable = Variable(
            size=len(components),
            type=variable.type,
            lower_bound=variable.lower_bound[idx],
            upper_bound=variable.upper_bound[idx],
        )
        self.__name_to_variable[name] = new_variable
        self.__name_to_normalization_mask[name] = self.__compute_normalization_mask(
            new_variable
        )
        self.__reindex()
        self.bump_version()

    def __getitem__(self, name: str) -> Variable:
        try:
            return self.__name_to_variable[name]
        except KeyError:
            msg = f"No variable named {name!r}."
            raise UnknownVariableError(msg) from None

    def get_integer_mask(self) -> BooleanArray:
        """Return whether the components of the full vector are integer.

        Returns:
            Whether the components of the full vector are integer
            (one result per component).
        """
        if not self.__name_to_variable:
            return zeros(0, dtype=bool)

        return concatenate(
            tuple(
                full(variable.size, variable.type == DataType.INTEGER, dtype=bool)
                for variable in self.__name_to_variable.values()
            )
        )

    @property
    def has_integer_variable(self) -> bool:
        """Whether the set has at least one integer variable."""
        return any(
            variable.type == DataType.INTEGER
            for variable in self.__name_to_variable.values()
        )

    def __compute_normalization_mask(self, variable: Variable) -> BooleanArray:
        """Compute the normalization policy mask of a variable.

        Unbounded variables (one or both bounds infinite) are not normalized.
        Integer variables are not normalized
        unless `enable_integer_variables_normalization` is `True`.

        Args:
            variable: The variable.

        Returns:
            The per-component normalization mask.
        """
        if (
            variable.type == DataType.FLOAT
            or self.__enable_integer_variables_normalization
        ):
            return logical_and(
                variable.lower_bound != -inf, variable.upper_bound != inf
            )
        return full(variable.size, False)

    def __iter__(self) -> Iterator[str]:
        return iter(self.__name_to_variable)

    def __len__(self) -> int:
        return len(self.__name_to_variable)
