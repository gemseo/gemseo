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
from typing import Literal

from numpy import array
from numpy import asarray
from numpy import atleast_1d
from numpy import float64
from numpy import isfinite
from numpy import unique
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator

from gemseo.space.variable._formatting import format_components
from gemseo.space.variable.base import _LOWER_BOUND
from gemseo.space.variable.base import _UPPER_BOUND
from gemseo.space.variable.base import BaseVariable
from gemseo.space.variable.base import DataType
from gemseo.space.variable.base import ScalarBoundType
from gemseo.util.pydantic_ndarray import NDArrayPydantic
from gemseo.util.string import pretty_str
from gemseo.util.typing import BooleanArray

_CHOICES: Final[str] = "choices"
"""The name of the field storing the choices of a discrete variable."""

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any

    from typing_extensions import Self

    from gemseo.util.typing import NumberArray
    from gemseo.util.typing import RealArray

_MAX_DISPLAYED_CHOICES: Final[int] = 6
"""The number of choices displayed in an error message."""

ChoicesType = (
    NDArrayPydantic[int]
    | NDArrayPydantic[float]
    | list[ScalarBoundType]
    | set[ScalarBoundType]
    | tuple[ScalarBoundType, ...]
)


class DiscreteVariable(BaseVariable):
    r"""A scalar discrete variable.

    Its domain is a finite set of numeric values that cannot be changed.
    The smallest (resp. largest) value
    defines the lower (resp. upper) bound of the variable.
    """

    type: ClassVar[DataType.DISCRETE] = DataType.DISCRETE

    size: Literal[1] = 1

    choices: ChoicesType = Field(
        description="""The values that the variable can take.

    They are stored without duplication and in ascending order.
    """
    )

    _NORMALIZATION_MASK: Final[BooleanArray] = array([False])
    """The normalization mask."""

    @field_validator("size", mode="before")
    @classmethod
    def __check_size(cls, value: Any) -> Any:
        """Check that the size is 1.

        Reject a size other than 1 here with a clear message,
        instead of leaving Pydantic reject it with a message naming the
        internal `size` field,
        e.g. when a caller asks for more components than the single one
        a discrete variable has,
        such as
        [DesignSpace.filter_dimensions][gemseo.space.design.DesignSpace.filter_dimensions]
        with duplicated indices.

        Args:
            value: The size passed by the caller.

        Returns:
            The value 1.

        Raises:
            ValueError: If the size is not 1.
        """
        if value != 1:
            msg = f"A discrete variable is scalar; its size cannot be set to {value}."
            raise ValueError(msg)

        return 1

    @model_validator(mode="after")
    def __validate_discrete_variable(self) -> Self:
        """Validate the variable and derive its bounds from its choices.

        Returns:
            The instance.

        Raises:
            ValueError: If a bound is set explicitly,
                or if the choices are not one-dimensional,
                or if there is no choice,
                or if some choices are not finite numbers.
        """
        self.__check_bounds_are_not_set()
        choices = self.__cast_choices_to_new_array()
        self.__check_choices(choices)
        sorted_unique_choices = unique(choices)
        # Freeze the validated choices using setflags + view.
        sorted_unique_choices.setflags(write=False)
        # Bypass assignment validation to avoid recursion when using setattr.
        self.__dict__[_CHOICES] = sorted_unique_choices.view()
        self.__derive_bounds(sorted_unique_choices)
        return self

    def __check_bounds_are_not_set(self) -> None:
        """Check that no bound is passed explicitly.

        Raises:
            ValueError: If a bound is passed explicitly.
        """
        fields_set = self.model_fields_set
        if fields_set & {_LOWER_BOUND, _UPPER_BOUND}:
            msg = (
                "The domain of a discrete variable is its choices, "
                "from which its bounds are derived; the bounds are not settable."
            )
            raise ValueError(msg)

    def __cast_choices_to_new_array(self) -> RealArray:
        """Cast the choices to a new NumPy array.

        The field type already guarantees
        that the choices are numbers,
        so the conversion cannot fail.

        Returns:
            The choices as a new NumPy array.
        """
        values = atleast_1d(asarray(self.choices, dtype=float64))
        # Copy so that freezing below does not affect an array owned by the caller.
        return values.copy()

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

    def __derive_bounds(self, choices: RealArray) -> None:
        """Derive the bounds from the sorted choices.

        Args:
            choices: The choices, sorted in ascending order.
        """
        for name, index in ((_LOWER_BOUND, 0), (_UPPER_BOUND, -1)):
            bound = array([choices[index]], dtype=self.component_type)
            bound.setflags(write=False)
            self.__dict__[name] = bound.view()

    def compute_normalization_mask(  # noqa: D102
        self, enable_integer_normalization: bool
    ) -> BooleanArray:
        return self._NORMALIZATION_MASK

    def find_components_outside_domain(self, choice: NumberArray) -> set[int]:  # noqa: D102
        choice_0 = atleast_1d(choice)[0]
        if choice_0 is None:
            return set()

        if (self.choices == choice_0).any():
            return set()

        return {0}

    def compute_default_value(self) -> NumberArray:
        """
        Returns:
            The lowest choice.
        """  # noqa: D205, D212
        return array([self.choices[0]], dtype=self.component_type)

    def _format_choices(self) -> str:
        """Return a readable representation of the choices for an error message.

        Beyond `_MAX_DISPLAYED_CHOICES` choices,
        the set is elided around its extremes and followed by its length,
        so that a message quoting many choices stays readable.

        Returns:
            The representation of the choices,
            e.g. `"[2.0, 4.0, 6.0, ..., 96.0, 98.0, 100.0] (50 choices)"`.
        """
        choices = self.choices
        if len(choices) <= _MAX_DISPLAYED_CHOICES:
            return f"[{pretty_str(choices, sort=False, use_and=False)}]"

        n_head = _MAX_DISPLAYED_CHOICES // 2
        n_tail = _MAX_DISPLAYED_CHOICES - n_head
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
        # NumPy does not preserve the writeable flag across pickling,
        # and Pydantic restores the model without re-validating it,
        # so refreeze the choices here.
        choices = self.__dict__[_CHOICES]
        choices.setflags(write=False)
        self.__dict__[_CHOICES] = choices.view()
