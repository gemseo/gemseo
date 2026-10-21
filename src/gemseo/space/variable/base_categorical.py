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
"""The base class of the categorical variables."""

from __future__ import annotations

from abc import abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Final

from numpy import array
from numpy import asarray
from numpy import atleast_1d
from numpy import int64
from numpy import isfinite
from numpy import isreal

from gemseo.space.variable._formatting import format_components
from gemseo.space.variable.base import CoordinateDType
from gemseo.space.variable.deterministic import BaseDeterministicVariable
from gemseo.util._numpy import freeze_array
from gemseo.util.string import pretty_repr

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence
    from typing import Any
    from typing import Self

    from numpy import ndarray

    from gemseo.util.typing import BooleanArray
    from gemseo.util.typing import IntegerArray
    from gemseo.util.typing import NumberArray

_normalization_masks: Final[tuple[BooleanArray, BooleanArray]] = (
    freeze_array(array([False])),
    freeze_array(array([True])),
)
"""The normalization masks of a categorical variable (read-only).

The coordinate of a categorical variable is an integer,
so it is normalized whenever the integer variables are:
the first mask applies when they are not normalized,
the second one when they are.
"""


class BaseCategoricalVariable(BaseDeterministicVariable):
    r"""The base class of a scalar categorical variable.

    Its domain is a finite set of unordered labels, also called categories,
    e.g. materials.
    The labels are strings, without duplication;
    a subclass defines them.

    A categorical variable is neither bounded nor castable to a NumPy type.
    A discipline reads its value as a label,
    while a design vector stores it as the position of this label
    among the labels, starting from zero, called its coordinate.

    A categorical variable has no derivative.
    The gradient of a function of a design vector
    stores zeros at the coordinates of its categorical variables,
    as for the variables on which this function does not depend.
    """

    coordinate_type: ClassVar[CoordinateDType] = int64
    """The NumPy type of the coordinate of the variable.

    The coordinate of a label is its position among the labels.
    """

    if TYPE_CHECKING:
        # This member is declared for static typing only;
        # see [BaseVariable][gemseo.space.variable.base.BaseVariable].

        @property
        def labels(self) -> tuple[str, ...]:
            """The labels that the variable can take.

            The position of a label is its index in this tuple.
            """

    @property
    def size(self) -> int:
        """The size of the variable."""
        return 1

    @cached_property
    def __label_to_position(self) -> dict[str, int]:
        """The position of each label."""
        return {label: position for position, label in enumerate(self.labels)}

    @cached_property
    def __label_array(self) -> ndarray:
        """The labels as an array, read by position."""
        return asarray(self.labels)

    def get_normalization_mask(  # noqa: D102
        self, enable_integer_normalization: bool
    ) -> BooleanArray:
        # A view, so that reassigning the shape, the strides or the data type
        # of the mask handed out, which NumPy allows on a read-only array,
        # cannot reach the mask shared by every categorical variable.
        return _normalization_masks[bool(enable_integer_normalization)].view()

    def find_components_outside_domain(self, coordinate: NumberArray) -> set[int]:  # noqa: D102
        coordinate_0 = atleast_1d(coordinate)[0]
        if coordinate_0 is None:
            return set()

        # A label, or any other non-numeric value, is never a coordinate;
        # return early to avoid the raw numpy error that
        # `.real` or `isfinite` would otherwise raise on it.
        if not isreal(coordinate_0):
            return {0}

        coordinate_0 = coordinate_0.real
        # An infinite or NaN coordinate is never the position of a label;
        # return early to avoid the RuntimeWarning that `%` would emit on it.
        if not isfinite(coordinate_0):
            return {0}

        if 0 <= coordinate_0 <= len(self.labels) - 1 and coordinate_0 % 1 == 0:
            return set()

        return {0}

    def get_default_value(self) -> NumberArray:
        """
        Returns:
            The coordinate of the first label, namely zero.
        """  # noqa: D205, D212
        return array([0], dtype=self.coordinate_type)

    def filter_components(self, components: Sequence[int]) -> Self:  # noqa: D102
        return self._filter_scalar_components(components)

    def encode(self, labels: Any) -> IntegerArray:
        """Convert labels into coordinates.

        Args:
            labels: One label or an iterable of labels of the variable.
                A value that is not a label, whatever its type, is rejected.

        Returns:
            The position of each label among the labels.

        Raises:
            ValueError: If a value is not one of the labels.
        """
        if isinstance(labels, str) or not hasattr(labels, "__iter__"):
            labels = [labels]

        label_to_position = self.__label_to_position
        positions = []
        for label in labels:
            try:
                positions.append(label_to_position[label])
            except (KeyError, TypeError):
                msg = (
                    f"{label!r} is not a label of the variable; "
                    f"the labels are {self._format_labels()}."
                )
                raise ValueError(msg) from None

        return array(positions, dtype=self.coordinate_type)

    def decode(self, coordinates: NumberArray) -> ndarray:
        """Convert coordinates into labels.

        Args:
            coordinates: The position of each label among the labels.

        Returns:
            The labels, with the shape of the coordinates.

        Raises:
            ValueError: If a coordinate is not the position of a label.
        """
        coordinates = asarray(coordinates)
        inside = (
            (coordinates >= 0)
            & (coordinates <= len(self.labels) - 1)
            & (coordinates % 1 == 0)
        )
        # A design vector stores the positions as floats,
        # which are displayed as integers when they are integers.
        outside = [
            int(coordinate) if coordinate % 1 == 0 else coordinate
            for coordinate in coordinates[~inside].tolist()
        ]
        if outside:
            msg = (
                f"{pretty_repr(outside)} "
                f"{'are' if len(outside) > 1 else 'is'} not the position "
                f"of a label; the labels are {self._format_labels()}."
            )
            raise ValueError(msg)

        return self.__label_array[coordinates.astype(self.coordinate_type)]

    @abstractmethod
    def _format_labels(self) -> str:
        """Return a readable representation of the labels for an error message.

        Returns:
            The representation of the labels.
        """

    def _get_out_of_domain_message(
        self, name: str, value: NumberArray, indices: Iterable[int]
    ) -> str:
        # A categorical variable is scalar, so its only caller,
        # `check_addable_value`, can never pass more than the single component
        # `find_components_outside_domain` may return; no plural wording is needed.
        return (
            f"The following coordinate of variable '{name}' "
            f"is not the position of one of its labels "
            f"{self._format_labels()}: "
            f"{format_components(value, indices)}."
        )

    def _get_out_of_domain_component_message(
        self, name: str, index: int, value_i: Any
    ) -> str:
        return (
            f"The variable {name} is categorical "
            f"with the labels {self._format_labels()}; "
            f"got the coordinate {name}[{index}] = {value_i}, "
            f"which is not the position of a label."
        )
