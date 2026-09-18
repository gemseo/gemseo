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
"""Base deterministic variable."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import ClassVar

from gemseo.space.variable._formatting import format_components
from gemseo.space.variable.base import BaseVariable
from gemseo.space.variable.base import DataType
from gemseo.util.string import pretty_str

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence
    from typing import Any
    from typing import Self

    from gemseo.util.typing import BooleanArray
    from gemseo.util.typing import NumberArray


class BaseDeterministicVariable(BaseVariable, ABC):
    """The base class of a variable that is not random.

    A deterministic variable takes a value,
    which a space can check against the domain of the variable
    and an optimization algorithm can normalize,
    unlike a random variable, which is defined by a probability distribution.
    """

    type: ClassVar[DataType] = DataType.FLOAT
    """The type of data."""

    @abstractmethod
    def get_normalization_mask(
        self, enable_integer_normalization: bool
    ) -> BooleanArray:
        """Return the per-component normalization policy of the variable.

        The mask is read-only and may be shared between variables;
        copy it before modifying it.

        Args:
            enable_integer_normalization: Whether to normalize the integer variables.

        Returns:
            Whether the components of the variable are normalized
            (one result per component);
            this array is read-only.
        """

    @abstractmethod
    def get_default_value(self) -> NumberArray:
        """Return the default value of the variable.

        Returns:
            The default value of the variable.
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

    def _filter_scalar_components(self, components: Sequence[int]) -> Self:
        """Return a scalar variable restricted to some of its components.

        A scalar variable has a single component,
        so the only selection it can honor is that component alone.

        Args:
            components: The components to be kept.

        Returns:
            The variable itself.

        Raises:
            ValueError: If a component to be kept does not exist,
                or if the components are not the single component of the variable.
        """
        indices = list(components)
        # Asking for a component that the variable does not have
        # is a different mistake from asking for a size that it cannot take,
        # so it is reported as such.
        if nonexistent_components := {index for index in indices if index != 0}:
            msg = (
                f"A {self.type} variable is scalar, so its only component is 0; "
                f"got {pretty_str(nonexistent_components)}."
            )
            raise ValueError(msg)

        if indices != [0]:
            msg = (
                f"A {self.type} variable is scalar; "
                f"its size cannot be set to {len(indices)}."
            )
            raise ValueError(msg)

        return self

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
