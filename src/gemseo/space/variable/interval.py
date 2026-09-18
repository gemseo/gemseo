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
"""Base variable whose domain is an interval."""

from __future__ import annotations

from abc import ABC
from itertools import starmap
from numbers import Real
from typing import TYPE_CHECKING
from typing import Final

from numpy import array
from numpy import atleast_1d
from numpy import full
from numpy import inf
from numpy import isfinite
from numpy import isnan
from numpy import ndarray
from pydantic import Field
from pydantic import PositiveInt
from pydantic import model_validator

from gemseo.space.variable._formatting import format_components
from gemseo.space.variable._view_field import expose_fields_as_views
from gemseo.space.variable.deterministic import BaseDeterministicVariable
from gemseo.space.variable.numeric import BaseNumericVariable
from gemseo.space.variable.numeric import BoundType
from gemseo.util._numpy import freeze_array

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any
    from typing import Self

    from gemseo.util.typing import NumberArray

_lower_bound: Final[str] = "lower_bound"
"""The tag for the lower bound."""

_upper_bound: Final[str] = "upper_bound"
"""The tag for the upper bound."""


class BaseIntervalVariable(BaseDeterministicVariable, BaseNumericVariable, ABC):
    """The base class of a variable whose domain is an interval.

    Such a variable is defined by
    a size,
    a data type
    and the bounds of its components,
    which the caller supplies.

    When `size > 1`,
    a bound could be defined with a scalar,
    in that case the bound will be converted to a NumPy array of the expected `size`.
    The components of a bound are stored
    with the NumPy type of the components of the variable,
    whatever the type of the value the caller supplies,
    unless a component is infinite,
    which an integer type cannot hold.

    The bounds are read as read-only arrays,
    handed out as views of what the variable stores,
    so that reassigning the shape, the strides or the data type of a bound,
    which NumPy allows on a read-only array,
    changes that view only.

    This class is abstract:
    a concrete subclass pins
    [type][gemseo.space.variable.deterministic.BaseDeterministicVariable.type]
    to one [DataType][gemseo.space.variable.base.DataType] member
    and implements the kind-specific hooks.
    """

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
            self.__cast_bound(name)

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
        # Read the value the caller supplied from the __dict__,
        # as the field is read as a view once it holds an array.
        bound = self.__dict__[bound_name]

        if isinstance(bound, Real):
            bound = full(self.size, bound)
        elif not isinstance(bound, ndarray):
            bound = atleast_1d(bound)

        # Freeze so that mutating a bound cannot bypass the version bump
        # and leave the derived caches serving stale bounds.
        # Bypass assignment validation to avoid recursion when using setattr.
        self.__dict__[bound_name] = freeze_array(bound)

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
        bound = self.__dict__[bound_name]

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

    def __cast_bound(
        self,
        bound_name: str,
    ) -> None:
        """Cast a bound to the NumPy type of the components of the variable.

        The cast is done after the bound has been checked,
        so that a component outside the domain of the variable is rejected
        instead of being silently cast into it,
        e.g. a non-integer bound of an integer variable
        or a bound too large for the type of the components.

        Args:
            bound_name: The name of the bound.
        """
        bound = self.__dict__[bound_name]
        # An infinite component cannot be cast to an integer,
        # so a bound with one keeps the type it was supplied with
        # and the other components rely on this value.
        if bound.dtype == self.component_type or not isfinite(bound).all():
            return

        # Bypass assignment validation to avoid recursion when using setattr.
        self.__dict__[bound_name] = freeze_array(bound.astype(self.component_type))

    @staticmethod
    def get_default_component_value(
        lower_bound_i: float, upper_bound_i: float
    ) -> float:
        """Return the default value of a component from its bounds.

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

    def get_default_value(self) -> NumberArray:
        """
        Returns:
            The component-wise center.
        """  # noqa: D205, D212
        return array(
            list(
                starmap(
                    self.get_default_component_value,
                    zip(self.lower_bound, self.upper_bound, strict=True),
                )
            ),
            dtype=self.component_type,
        )

    def filter_components(self, components: Sequence[int]) -> Self:  # noqa: D102
        indices = list(components)
        if indices == list(range(self.size)):
            # Keeping every component in order is an identity;
            # the frozen variable can be shared.
            return self

        return self.model_copy(
            update={
                "size": len(indices),
                _lower_bound: self.lower_bound[indices],
                _upper_bound: self.upper_bound[indices],
            }
        )

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        # Pickling preserves neither the writeable flag nor the base, so refreeze.
        for name in (_lower_bound, _upper_bound):
            self.__dict__[name] = freeze_array(self.__dict__[name])


# A property could not take the name of a field,
# so the bounds are read as views of the frozen arrays stored
# through descriptors set once pydantic has built the class.
expose_fields_as_views(BaseIntervalVariable, _lower_bound, _upper_bound)
