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
"""Variable."""

from __future__ import annotations

from numbers import Real
from typing import TYPE_CHECKING
from typing import Final

from numpy import atleast_1d
from numpy import float64
from numpy import full
from numpy import inf
from numpy import int64
from numpy import isfinite
from numpy import isnan
from numpy import logical_and
from numpy import mod
from numpy import ndarray
from pydantic import BaseModel
from pydantic import PositiveInt
from pydantic import model_validator
from strenum import StrEnum

from gemseo.util.pydantic_ndarray import NDArrayPydantic
from gemseo.util.string import pretty_str
from gemseo.util.typing import IntegerArray
from gemseo.util.typing import RealArray

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping
    from typing import Any

    from typing_extensions import Self

_LOWER_BOUND: Final[str] = "lower_bound"
"""The tag for the lower bound."""

_UPPER_BOUND: Final[str] = "upper_bound"
"""The tag for the upper bound."""

ScalarBoundType = int | float
BoundType = (
    NDArrayPydantic[int]
    | NDArrayPydantic[float]
    | list[ScalarBoundType]
    | tuple[ScalarBoundType]
    | ScalarBoundType
)
BoundArray = IntegerArray | RealArray


def format_components(array: ndarray, indices: Iterable[int]) -> str:
    """Return a readable representation of some components of an array.

    Args:
        array: The array.
        indices: The indices of the components,
            sorted in ascending order in the representation.

    Returns:
        The components with their indices,
        e.g. `"nan (index 0) and inf (index 2)"`.
    """
    return pretty_str(
        [f"{array[index]} (index {index})" for index in sorted(indices)], sort=False
    )


class DataType(StrEnum):
    """The type of variable data."""

    FLOAT = "float"
    INTEGER = "integer"


# The mapping from a variable data type to a numpy type,
# this is defined at the module level because pydantic does not allow class attributes
# that are dictionary.
TYPE_MAP: Final[dict[str, type[int64 | float64]]] = {
    DataType.INTEGER: int64,
    DataType.FLOAT: float64,
}


class Variable(BaseModel, frozen=True):
    """A variable.

    A variable is defined by
    a size,
    a data type
    and the bounds of its components.

    When `size > 1`,
    a bound could be defined with a scalar,
    in that case the bound will be converted to a NumPy array of the expected `size`.

    A variable is immutable.
    """

    size: PositiveInt = 1
    """The size of the variable."""

    type: DataType = DataType.FLOAT
    """The type of data."""

    lower_bound: BoundType = -inf
    """The lower bound of the variable."""

    upper_bound: BoundType = inf
    """The upper bound of the variable."""

    @model_validator(mode="after")
    def __validate_variable(self) -> Self:
        """Validate the variable.

        Returns:
            The instance.
        """
        for name in (_LOWER_BOUND, _UPPER_BOUND):
            self.__convert_bound(name)
            self.__check_bound(name)

        if (self.upper_bound < self.lower_bound).any():
            msg = "The upper bounds must be greater than or equal to the lower bounds."
            raise ValueError(msg)

        return self

    def __convert_bound(
        self,
        bound_name: str,
    ) -> None:
        r"""Convert a scalar bound to a NumPy array one.

        Args:
            bound_name: The name of the bound.
        """
        bound = getattr(self, bound_name)

        if isinstance(bound, ndarray):
            # Copy so that freezing below does not affect the array owned by the
            # caller.
            bound = bound.copy()
        elif isinstance(bound, Real):
            # inf cannot be cast to int and other components rely on this value.
            dtype = None if bound in (-inf, inf) else TYPE_MAP[self.type]
            bound = full(self.size, bound, dtype=dtype)
        else:
            bound = atleast_1d(bound)

        # Freeze the validated bound array so that an accidental in-place mutation
        # cannot bypass the version bump
        # and leave the derived caches serving stale bounds.
        # The accessors of Bounds hand out read-only views of this array,
        # so that its writeable flag cannot be re-enabled from the outside.
        bound.setflags(write=False)

        # Bypass assignment validation to avoid recursion when using setattr.
        self.__dict__[bound_name] = bound

    def __check_bound(
        self,
        bound_name: str,
    ) -> None:
        """Check a bound.

        Args:
            bound_name: The name of the bound.

        Raises:
            ValueError:
                If the bound is not one-dimensional,
                of if the bound does not have the right size,
                or if some bound components are not numbers,
                or if the variable is of integer type
                and has some finite non-integer components.
        """
        bound = getattr(self, bound_name)

        bound_prefix = bound_name.split("_")[0]

        if len(bound.shape) > 1:
            msg = f"The {bound_prefix} bound has a dimension greater than 1."
            raise ValueError(msg)

        if bound.size != self.size:
            msg = f"The {bound_prefix} bound should be of size {self.size}."
            raise ValueError(msg)

        # Check whether the components of the bound are numbers.
        indices = isnan(bound).nonzero()[0]
        if len(indices):
            plural = len(indices) > 1
            msg = (
                f"The following {bound_prefix} bound component"
                f"{'s are not numbers' if plural else ' is not a number'}: "
                f"{format_components(bound, indices)}."
            )
            raise ValueError(msg)

        if self.type == DataType.INTEGER:
            # Check whether the components of the bound are integers (or infinite).
            indices = logical_and(isfinite(bound), mod(bound, 1)).nonzero()[0]
            if len(indices):
                plural = len(indices) > 1
                msg = (
                    f"The following {bound_prefix} bound component"
                    f"{'s are' if plural else ' is'} neither integer nor infinite "
                    "while the variable is of type integer: "
                    f"{format_components(bound, indices)}."
                )
                raise ValueError(msg)

    def __copy__(self) -> Self:
        # A variable is immutable and its bound arrays are read-only,
        # so a copy can be shared with the original.
        # This also keeps the bound arrays frozen,
        # since NumPy does not preserve the writeable flag across a copy.
        return self

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Self:
        return self

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> Self:
        """Return a copy of the variable, updated with new field values.

        Args:
            update: The new field values, if any.
            deep: Whether to deep-copy the variable;
                this has no effect since the bound arrays are copied by the validation.

        Returns:
            The variable itself without an update, otherwise a new variable.
        """
        # The base implementation writes the update into the __dict__ of the object
        # returned by __copy__/__deepcopy__, which is this very instance;
        # rebuild through validation instead, so that the original is left alone
        # and the new bounds are converted, checked and frozen.
        if not update:
            return self

        return self.model_validate({**self.__dict__, **update})

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        # NumPy does not preserve the writeable flag across pickling,
        # and pydantic restores the model without re-validating it,
        # so refreeze the bound arrays here.
        for name in (_LOWER_BOUND, _UPPER_BOUND):
            self.__dict__[name].setflags(write=False)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, self.__class__)
            and self.size == other.size
            and self.type == other.type
            and (self.lower_bound == other.lower_bound).all()
            and (self.upper_bound == other.upper_bound).all()
        )
