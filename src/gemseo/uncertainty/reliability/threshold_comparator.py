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
"""Threshold comparator."""

from __future__ import annotations

import operator
from enum import StrEnum
from typing import TYPE_CHECKING
from typing import Final

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import TypeAlias

    from gemseo.util.typing import BooleanArray
    from gemseo.util.typing import RealArray

    _ComparisonOperator: TypeAlias = Callable[[RealArray, float], BooleanArray]
    """The signature of an element-wise comparison operator."""


class ThresholdComparator(StrEnum):
    """A comparison between a variable of interest and a threshold.

    It carries the symbol used for string rendering,
    the element-wise NumPy operator used for evaluation
    and the complementary comparison used for negation.
    """

    LESS = "<"
    """The variable of interest is strictly less than the threshold."""

    LESS_EQUAL = "<="
    """The variable of interest is less than or equal to the threshold."""

    GREATER = ">"
    """The variable of interest is strictly greater than the threshold."""

    GREATER_EQUAL = ">="
    """The variable of interest is greater than or equal to the threshold."""

    @property
    def complement(self) -> ThresholdComparator:
        """The complementary comparison, e.g. `<=` for `>`.

        The complement is exact for non-NaN values only:
        a NaN value satisfies neither a comparison nor its complement.
        """
        return _comparator_to_complement[self]

    def compare(self, values: RealArray, threshold: float) -> BooleanArray:
        """Compare an array of values to a threshold.

        Args:
            values: The values of the variable of interest.
            threshold: The threshold.

        Returns:
            The boolean indicator of the comparison, element-wise.
        """
        return _comparator_to_operator[self](values, threshold)


_comparator_to_complement: Final[dict[ThresholdComparator, ThresholdComparator]] = {
    ThresholdComparator.LESS: ThresholdComparator.GREATER_EQUAL,
    ThresholdComparator.LESS_EQUAL: ThresholdComparator.GREATER,
    ThresholdComparator.GREATER: ThresholdComparator.LESS_EQUAL,
    ThresholdComparator.GREATER_EQUAL: ThresholdComparator.LESS,
}
"""The map from a comparator to its complement."""

_comparator_to_operator: Final[dict[ThresholdComparator, _ComparisonOperator]] = {
    ThresholdComparator.LESS: operator.lt,
    ThresholdComparator.LESS_EQUAL: operator.le,
    ThresholdComparator.GREATER: operator.gt,
    ThresholdComparator.GREATER_EQUAL: operator.ge,
}
"""The map from a comparator to its element-wise NumPy operator."""
