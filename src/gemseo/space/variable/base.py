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
"""Base variable."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from collections.abc import Mapping
from enum import StrEnum
from typing import TYPE_CHECKING

from numpy import ndarray
from pydantic import BaseModel
from pydantic import model_validator

from gemseo.util.string import pretty_str

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any
    from typing import Self


class DataType(StrEnum):
    """The type of variable data."""

    DISCRETE = "discrete"
    FLOAT = "float"
    INTEGER = "integer"


class BaseVariable(BaseModel, ABC, frozen=True, extra="forbid"):
    """The base class of a variable.

    Whatever its kind, a variable reads as
    `size`, the number of its components,
    and `type`, the [DataType][gemseo.space.variable.base.DataType] of these
    components;
    what else defines it depends on its kind,
    e.g. bounds for a variable whose components are numbers
    and probability distributions for a random variable.

    A variable is immutable.
    Whatever a kind of variable takes as input is a field of the model,
    and whatever derives from these inputs is a read-only property,
    so that the fields of a variable are exactly what defines it;
    passing such a derived member raises a `ValueError` naming it.
    """

    if TYPE_CHECKING:
        # These members are declared for static typing only.
        # A property here would be a data descriptor
        # shadowing the field of a subclass storing the value,
        # since pydantic keeps the values of the fields in the __dict__ of the model,
        # and a field here could not be overridden by a property
        # in a subclass deriving the value.

        @property
        def size(self) -> int:
            """The size of the variable."""

        @property
        def type(self) -> DataType:
            """The type of data."""

    @model_validator(mode="before")
    @classmethod
    def __check_inputs(cls, data: Any) -> Any:
        """Reject a member of the variable that does not define it.

        What defines a variable are its fields;
        a member deriving from them, e.g. the size of a scalar variable,
        is read-only,
        and a caller passing one gets told so
        instead of being told that the input is not permitted.

        Args:
            data: The arguments passed by the caller.

        Returns:
            The arguments passed by the caller.

        Raises:
            ValueError: If an argument is a member deriving from the fields.
        """
        if not isinstance(data, Mapping):
            return data

        field_names = cls.model_fields
        derived_names = [
            name for name in data if name not in field_names and hasattr(cls, name)
        ]
        if derived_names:
            several_names = len(derived_names) > 1
            several_fields = len(field_names) > 1
            msg = (
                f"{pretty_str(derived_names, use_and=True)} "
                f"{'are' if several_names else 'is'} read-only; "
                f"the {'inputs' if several_fields else 'input'} of a "
                f"{cls.__name__} {'are' if several_fields else 'is'} "
                f"{pretty_str(tuple(field_names), use_and=True)}."
            )
            raise ValueError(msg)

        return data

    @abstractmethod
    def filter_components(self, components: Sequence[int]) -> Self:
        """Return the variable restricted to some of its components.

        Args:
            components: The components to be kept.

        Returns:
            The variable itself when every component is kept in order,
            otherwise a new variable defined by the components to be kept.
        """

    def __copy__(self) -> Self:
        # A variable is immutable and the arrays it hands out are frozen,
        # so a copy can be shared with the original;
        # this also keeps these arrays frozen, which copying would not.
        return self

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Self:
        return self

    def _get_field_values(self) -> dict[str, Any]:
        """Return the values of the fields of the variable.

        The __dict__ of the model also carries the caches of its cached properties,
        e.g. the joint probability distribution of a random variable,
        which are not part of what defines the variable.

        Returns:
            The values of the fields of the variable.
        """
        field_names = type(self).model_fields
        return {
            name: value for name, value in self.__dict__.items() if name in field_names
        }

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> Self:
        """Return a copy of the variable, updated with new field values.

        Args:
            update: The new field values, if any.
            deep: Whether to deep-copy the variable;
                this has no effect since a variable is immutable.

        Returns:
            The variable itself without an update, otherwise a new variable.
        """
        # The base implementation writes the update into the __dict__ of the object
        # returned by __copy__/__deepcopy__, which is this very instance,
        # and would leave the cache of a cached property stale;
        # rebuild through validation instead, so that the original is left alone
        # and the new values are converted, checked and frozen.
        if not update:
            return self

        return self.model_validate({**self._get_field_values(), **update})

    def __getstate__(self) -> dict[str, Any]:
        state = dict(super().__getstate__())
        # Pydantic pickles the __dict__ of the model as is;
        # drop the caches of the cached properties from it
        # so that the pickle carries the fields only,
        # e.g. so that the pickle of a random variable carries its settings
        # instead of an object graph of a third-party library.
        state["__dict__"] = self._get_field_values()
        return state

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseVariable):
            return False

        if self.type != other.type:
            return False

        # Compare the fields of the kind of the variable
        # so that a field added by a subclass takes part in the comparison.
        # The declaration order is preserved so that size is compared before the bounds,
        # whose comparison would raise on arrays of different sizes.
        names = dict.fromkeys((*type(self).model_fields, *type(other).model_fields))
        for name in names:
            if not (hasattr(self, name) and hasattr(other, name)):
                # A field declared by only one of the two kinds.
                return False

            self_value = getattr(self, name)
            other_value = getattr(other, name)
            if (
                isinstance(self_value, ndarray)
                and isinstance(other_value, ndarray)
                and self_value.shape != other_value.shape
            ):
                return False

            comparison = self_value == other_value
            if isinstance(comparison, ndarray):
                if not comparison.all():
                    return False
            elif not comparison:
                return False

        return True
