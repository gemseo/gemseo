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
from itertools import starmap
from numbers import Real
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Final

from numpy import array
from numpy import atleast_1d
from numpy import float64
from numpy import full
from numpy import inf
from numpy import int64
from numpy import isnan
from numpy import ndarray
from pydantic import BaseModel
from pydantic import Field
from pydantic import PositiveInt
from pydantic import model_validator
from strenum import StrEnum

from gemseo.space.variable._formatting import format_components
from gemseo.util.pydantic_ndarray import NDArrayPydantic
from gemseo.util.typing import IntegerArray
from gemseo.util.typing import RealArray

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping
    from typing import Any

    from typing_extensions import Self

    from gemseo.util.typing import BooleanArray
    from gemseo.util.typing import NumberArray


_lower_bound: Final[str] = "lower_bound"
"""The tag for the lower bound."""

_upper_bound: Final[str] = "upper_bound"
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

# The `type` field of a variable shadows the builtin inside the class body,
# so the NumPy type of the components is aliased here.
ComponentDType = type[int64 | float64]


class DataType(StrEnum):
    """The type of variable data."""

    DISCRETE = "discrete"
    FLOAT = "float"
    INTEGER = "integer"


class BaseVariable(BaseModel, ABC, frozen=True, extra="forbid"):
    """The base class of a variable.

    A variable is defined by
    a size,
    a data type
    and the bounds of its components.

    When `size > 1`,
    a bound could be defined with a scalar,
    in that case the bound will be converted to a NumPy array of the expected `size`.

    A variable is immutable.

    This class is abstract:
    a concrete subclass pins [type][gemseo.space.variable.BaseVariable.type]
    to one [DataType][gemseo.space.variable.DataType] member
    and implements the kind-specific hooks.
    Build a variable with
    `VariableFactory`.
    """

    component_type: ClassVar[ComponentDType] = float64
    """The NumPy type of the components of the variable."""

    type: ClassVar[DataType] = DataType.FLOAT
    """The type of data."""

    size: PositiveInt = Field(default=1, description="The size of the variable.")

    lower_bound: BoundType = Field(
        default=-inf, description="The lower bound of the variable."
    )

    upper_bound: BoundType = Field(
        default=inf, description="The upper bound of the variable."
    )

    @model_validator(mode="after")
    def __validate_variable(self) -> Self:
        """Validate the variable.

        Returns:
            The instance.
        """
        for name in (_lower_bound, _upper_bound):
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
            dtype = None if bound in (-inf, inf) else self.component_type
            bound = full(self.size, bound, dtype=dtype)
        else:
            bound = atleast_1d(bound)

        # Freeze the converted bound array and store a read-only view of it:
        # an in-place mutation then raises
        # instead of bypassing the version bump
        # and leaving the derived caches serving stale bounds.
        # Freezing the array alone would not be enough,
        # since a caller can re-enable the writeable flag of an array owning its data;
        # a view does not own its data, so NumPy refuses to re-enable its flag.
        bound.setflags(write=False)

        # Bypass assignment validation to avoid recursion when using setattr.
        self.__dict__[bound_name] = bound.view()

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
                or if some finite bound components are outside
                the domain of the kind of variable.
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

        self.check_finite_bound_components(bound, bound_prefix)

    def cast(self, value: ndarray) -> NumberArray:
        """Cast a value of the variable to the NumPy type of the variable.

        Args:
            value: The value of the variable.

        Returns:
            The cast value of the variable.
        """
        return value.astype(self.component_type)

    @staticmethod
    def compute_default_component_value(
        lower_bound_i: float, upper_bound_i: float
    ) -> float:
        """Compute the default value of a component from its bounds.

        Use the center of the bounds when both are finite,
        otherwise the finite bound,
        otherwise zero.

        Args:
            lower_bound_i: The lower bound of the component.
            upper_bound_i: The upper bound of the component.

        Returns:
            The default value of the component.
        """
        if lower_bound_i == -inf:
            return 0.0 if upper_bound_i == inf else upper_bound_i

        if upper_bound_i == inf:
            return lower_bound_i

        return (lower_bound_i + upper_bound_i) / 2

    def compute_default_value(self) -> NumberArray:
        """Compute the default value of the variable.

        Returns:
            The component-wise center.
        """
        return array(
            list(
                starmap(
                    self.compute_default_component_value,
                    zip(self.lower_bound, self.upper_bound, strict=True),
                )
            ),
            dtype=self.component_type,
        )

    @abstractmethod
    def compute_normalization_mask(
        self, enable_integer_normalization: bool
    ) -> BooleanArray:
        """Compute the per-component normalization policy of the variable.

        Args:
            enable_integer_normalization: Whether to normalize the integer variables.

        Returns:
            Whether the components of the variable are normalized
            (one result per component).
        """

    def check_finite_bound_components(
        self, bound: BoundArray, bound_prefix: str
    ) -> None:
        """Check that the finite components of a bound are in the domain.

        Any finite component is a valid bound unless a subclass restricts the domain.

        Args:
            bound: The bound.
            bound_prefix: The prefix naming the bound in a message,
                either `"lower"` or `"upper"`.

        Raises:
            ValueError: If some finite components of the bound are outside
                the domain of the kind of variable.
        """

    def find_components_outside_domain(self, value: NumberArray) -> set[int]:
        """Return the indices of the components outside the domain of the variable.

        Any component is in the domain unless a subclass restricts it.

        Args:
            value: The value of the variable.

        Returns:
            The indices of the components outside the domain of the variable.
        """
        return set()

    def _get_out_of_domain_message(
        self, name: str, value: NumberArray, indices: Iterable[int]
    ) -> str:
        """Return the message telling that some values are outside the domain.

        The wording belongs to the kind of the variable,
        so that a kind whose domain is not an interval can phrase its own failure.

        Args:
            name: The name of the variable.
            value: The value of the variable.
            indices: The indices of the components outside the domain.

        Returns:
            The message.
        """
        indices = list(indices)
        plural = len(indices) > 1
        return (
            f"The following value{'s' if plural else ''} of variable '{name}' "
            f"{'are' if plural else 'is'} neither None nor {self.type} "
            f"while variable '{name}' is of type {self.type}: "
            f"{format_components(value, indices)}."
        )

    def _get_out_of_domain_component_message(
        self, name: str, index: int, value_i: Any
    ) -> str:
        """Return the message telling that a component is outside the domain.

        The wording belongs to the kind of the variable,
        so that a kind whose domain is not an interval can phrase its own failure.

        Args:
            name: The name of the variable.
            index: The index of the component.
            value_i: The value of the component.

        Returns:
            The message.
        """
        return (
            f"The variable {name} is of type {self.type}; "
            f"got {name}[{index}] = {value_i}."
        )

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

        # A bound always sits in __dict__, even one derived rather than passed by the
        # caller (e.g. the bounds of a discrete variable, derived from its potential
        # values). Re-including such a derived bound here would mark it as set on the
        # re-validated copy, tripping a subclass check meant for a caller-supplied
        # bound (e.g. DiscreteVariable.__check_bounds_are_not_set). Carry a bound over
        # only when it was set on this instance, or when the caller's update sets it.
        fields_set = self.model_fields_set
        payload = {
            name: value
            for name, value in self.__dict__.items()
            if name not in (_lower_bound, _upper_bound) or name in fields_set
        }
        payload.update(update)
        return self.model_validate(payload)

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        # NumPy preserves neither the writeable flag nor the base across pickling,
        # and pydantic restores the model without re-validating it,
        # so refreeze the bound arrays here and store read-only views of them again.
        for name in (_lower_bound, _upper_bound):
            bound = self.__dict__[name]
            bound.setflags(write=False)
            self.__dict__[name] = bound.view()

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
