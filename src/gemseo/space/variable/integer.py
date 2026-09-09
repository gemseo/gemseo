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
"""Integer variable."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from numpy import array
from numpy import atleast_1d
from numpy import floor
from numpy import full
from numpy import inf
from numpy import int64
from numpy import isinf
from numpy import logical_and
from numpy import mod

from gemseo.space.variable._formatting import format_components
from gemseo.space.variable.base import BaseVariable
from gemseo.space.variable.base import ComponentDType
from gemseo.space.variable.base import DataType

if TYPE_CHECKING:
    from gemseo.space.variable.base import BoundArray
    from gemseo.util.typing import BooleanArray
    from gemseo.util.typing import NumberArray


def _get_integer_mask(value: NumberArray | complex) -> BooleanArray:
    """Return whether the components of an array are integer.

    `None` and infinite values will be interpreted as integers

    Args:
        value: Either the array
            or a number that the method will cast into a 1D array.

    Returns:
        Whether the components of the array are integer.
    """
    return array([x is None or isinf(x) or not mod(x, 1) for x in atleast_1d(value)])


def _find_non_integer_indices(value: NumberArray) -> set[int]:
    """Return the indices of the components that are not integer.

    `None` and infinite components are treated as integer.

    Args:
        value: The 1D array to inspect.

    Returns:
        The indices of the non-integer components.
    """
    return set(range(len(value))) - set(_get_integer_mask(value).nonzero()[0])


class IntegerVariable(BaseVariable):
    """A variable whose components are integers."""

    component_type: ClassVar[ComponentDType] = int64

    type: ClassVar[DataType] = DataType.INTEGER

    def compute_normalization_mask(  # noqa: D102
        self, enable_integer_normalization: bool
    ) -> BooleanArray:
        if enable_integer_normalization:
            return logical_and(self.lower_bound != -inf, self.upper_bound != inf)

        return full(self.size, False)

    def check_finite_bound_components(  # noqa: D102
        self, bound: BoundArray, bound_prefix: str
    ) -> None:
        # Check whether the components of the bound are integers (or infinite).
        # Floor leaves the infinite components unchanged,
        # unlike mod, which returns nan and emits a RuntimeWarning for them.
        indices = (bound != floor(bound)).nonzero()[0]
        if len(indices):
            plural = len(indices) > 1
            msg = (
                f"The following {bound_prefix} bound component"
                f"{'s are' if plural else ' is'} neither integer nor infinite "
                "while the variable is of type integer: "
                f"{format_components(bound, indices)}."
            )
            raise ValueError(msg)

    def find_components_outside_domain(self, value: NumberArray) -> set[int]:  # noqa: D102
        return _find_non_integer_indices(value)
