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
"""Discrete variable."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Final

from numpy import array
from numpy import asarray
from numpy import atleast_1d
from numpy import float64
from numpy import isfinite
from numpy import unique
from pydantic import Field
from pydantic import model_validator

from gemseo.space.variable._formatting import format_components
from gemseo.space.variable._view_field import expose_fields_as_views
from gemseo.space.variable.base import DataType
from gemseo.space.variable.deterministic import BaseDeterministicVariable
from gemseo.space.variable.numeric import BaseNumericVariable
from gemseo.space.variable.numeric import ScalarBoundType
from gemseo.util._numpy import freeze_array
from gemseo.util.pydantic_ndarray import NDArrayPydantic
from gemseo.util.string import pretty_str

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence
    from typing import Any
    from typing import Self

    from gemseo.space.variable.numeric import BoundArray
    from gemseo.util.typing import BooleanArray
    from gemseo.util.typing import NumberArray
    from gemseo.util.typing import RealArray

_choices: Final[str] = "choices"
"""The name of the field storing the choices of a discrete variable."""

_normalization_mask: Final[BooleanArray] = freeze_array(array([False]))
"""The normalization mask of a discrete variable (read-only).

A discrete variable is never normalized,
so this mask is the same for every discrete variable;
it is frozen once and for all,
and `DiscreteVariable.get_normalization_mask` hands out views of it.
"""

_max_displayed_choices: Final[int] = 6
"""The number of choices displayed in an error message."""

ChoicesType = (
    NDArrayPydantic[int]
    | NDArrayPydantic[float]
    | list[ScalarBoundType]
    | set[ScalarBoundType]
    | tuple[ScalarBoundType, ...]
)


class DiscreteVariable(BaseDeterministicVariable, BaseNumericVariable):
    r"""A scalar discrete variable.

    Its domain is a finite set of numeric values that cannot be changed.
    The smallest (resp. largest) value
    defines the lower (resp. upper) bound of the variable.

    The choices are the only input of the variable;
    its size, which is one, and its bounds derive from them and are read-only.
    The choices are read as a read-only array,
    handed out as a view of what the variable stores,
    so that reassigning its shape, its strides or its data type,
    which NumPy allows on a read-only array,
    changes that view only.
    """

    type: ClassVar[DataType] = DataType.DISCRETE

    choices: ChoicesType = Field(
        description="""The values that the variable can take.

    They are stored without duplication and in ascending order.
    """
    )

    @model_validator(mode="after")
    def __validate_discrete_variable(self) -> Self:
        """Validate the variable.

        Returns:
            The instance.

        Raises:
            ValueError: If the choices are not one-dimensional,
                or if there is no choice,
                or if some choices are not finite numbers.
        """
        choices = self.__cast_choices_to_array()
        self.__check_choices(choices)
        # Bypass assignment validation to avoid recursion when using setattr.
        self.__dict__[_choices] = freeze_array(unique(choices))
        return self

    def __cast_choices_to_array(self) -> RealArray:
        """Cast the choices to a NumPy array.

        The field type already guarantees
        that the choices are numbers,
        so the conversion cannot fail.

        Returns:
            The choices as a NumPy array.
        """
        # Read the value the caller supplied from the __dict__,
        # as the field is read as a view once it holds an array.
        return atleast_1d(asarray(self.__dict__[_choices], dtype=float64))

    @staticmethod
    def __check_choices(choices: RealArray) -> None:
        """Check the shape and the components of the choices.

        Args:
            choices: The choices.

        Raises:
            ValueError: If the choices are not one-dimensional,
                or if there is no choice,
                or if some choices are not finite numbers.
        """
        if (n_dim := choices.ndim) > 1:
            msg = f"The dimension of choices must be 1; got {n_dim}."
            raise ValueError(msg)

        if not choices.size:
            msg = "A discrete variable must have at least one choice."
            raise ValueError(msg)

        indices = (~isfinite(choices)).nonzero()[0]
        if len(indices):
            plural = len(indices) > 1
            msg = (
                f"The following choice{'s are' if plural else ' is'} "
                f"not a finite number: {format_components(choices, indices)}."
            )
            raise ValueError(msg)

    @property
    def size(self) -> int:
        """The size of the variable."""
        return 1

    @property
    def lower_bound(self) -> BoundArray:
        """The lower bound of the variable (read-only)."""
        return self.__create_bound(0)

    @property
    def upper_bound(self) -> BoundArray:
        """The upper bound of the variable (read-only)."""
        return self.__create_bound(-1)

    def __create_bound(self, index: int) -> BoundArray:
        """Create a bound of the variable from its choices.

        The bound array is built on demand and handed out read-only.

        Args:
            index: The index of the choice defining the bound,
                the choices being sorted in ascending order.

        Returns:
            The bound of the variable.
        """
        return freeze_array(array([self.choices[index]], dtype=self.component_type))

    def get_normalization_mask(  # noqa: D102
        self, enable_integer_normalization: bool
    ) -> BooleanArray:
        # A view, so that reassigning the shape, the strides or the data type
        # of the mask handed out, which NumPy allows on a read-only array,
        # cannot reach the mask shared by every discrete variable.
        return _normalization_mask.view()

    def find_components_outside_domain(self, choice: NumberArray) -> set[int]:  # noqa: D102
        choice_0 = atleast_1d(choice)[0]
        if choice_0 is None:
            return set()

        if (self.choices == choice_0).any():
            return set()

        return {0}

    def get_default_value(self) -> NumberArray:
        """
        Returns:
            The lowest choice.
        """  # noqa: D205, D212
        return array([self.choices[0]], dtype=self.component_type)

    def filter_components(self, components: Sequence[int]) -> Self:  # noqa: D102
        return self._filter_scalar_components(components)

    def _format_choices(self) -> str:
        """Return a readable representation of the choices for an error message.

        Beyond `_max_displayed_choices` choices,
        the set is elided around its extremes and followed by its length,
        so that a message quoting many choices stays readable.

        Returns:
            The representation of the choices,
            e.g. `"[2.0, 4.0, 6.0, ..., 96.0, 98.0, 100.0] (50 choices)"`.
        """
        choices = self.choices
        if len(choices) <= _max_displayed_choices:
            return f"[{pretty_str(choices, sort=False, use_and=False)}]"

        n_head = _max_displayed_choices // 2
        n_tail = _max_displayed_choices - n_head
        head = pretty_str(choices[:n_head], sort=False, use_and=False)
        tail = pretty_str(choices[-n_tail:], sort=False, use_and=False)
        return f"[{head}, ..., {tail}] ({len(choices)} choices)"

    def _get_out_of_domain_message(
        self, name: str, value: NumberArray, indices: Iterable[int]
    ) -> str:
        # A discrete variable is scalar, so its only caller,
        # `check_addable_value`, can never pass more than the single component
        # `find_components_outside_domain` may return; no plural wording is needed.
        return (
            f"The following value of variable '{name}' "
            f"is not among its choices "
            f"{self._format_choices()}: "
            f"{format_components(value, indices)}."
        )

    def _get_out_of_domain_component_message(
        self, name: str, index: int, value_i: Any
    ) -> str:
        return (
            f"The variable {name} is discrete "
            f"with the choices {self._format_choices()}; "
            f"got {name}[{index}] = {value_i}."
        )

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        # Pickling preserves neither the writeable flag nor the base, so refreeze.
        self.__dict__[_choices] = freeze_array(self.__dict__[_choices])


# A property could not take the name of a field,
# so the choices are read as a view of the frozen array stored
# through a descriptor set once pydantic has built the class.
expose_fields_as_views(DiscreteVariable, _choices)
