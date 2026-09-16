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
"""Base variable whose components are numbers."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING
from typing import ClassVar

from numpy import float64
from numpy import int64

from gemseo.space.variable.base import BaseVariable
from gemseo.util.pydantic_ndarray import NDArrayPydantic
from gemseo.util.typing import IntegerArray
from gemseo.util.typing import RealArray

if TYPE_CHECKING:
    from numpy import ndarray

    from gemseo.util.typing import NumberArray

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


class BaseNumericVariable(BaseVariable, ABC):
    """The base class of a variable whose components are numbers.

    The components of such a variable are ordered,
    so the variable is bounded and reads as `lower_bound` and `upper_bound`,
    either bounds supplied by the caller
    or bounds derived from what defines the variable,
    e.g. the values it can take or the support of its probability distribution.

    A variable whose components are not numbers, e.g. labels,
    is neither bounded nor castable to a NumPy type,
    and so does not derive from this class.
    """

    component_type: ClassVar[ComponentDType] = float64
    """The NumPy type of the components of the variable."""

    if TYPE_CHECKING:
        # These members are declared for static typing only;
        # see [BaseVariable][gemseo.space.variable.base.BaseVariable].

        @property
        def lower_bound(self) -> BoundArray:
            """The lower bound of the variable."""

        @property
        def upper_bound(self) -> BoundArray:
            """The upper bound of the variable."""

    def cast(self, value: ndarray) -> NumberArray:
        """Cast a value of the variable to the NumPy type of the variable.

        Args:
            value: The value of the variable.

        Returns:
            The cast value of the variable.
        """
        return value.astype(self.component_type)

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
