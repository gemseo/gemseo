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
from typing import Union

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

from gemseo.typing import IntegerArray
from gemseo.typing import RealArray
from gemseo.utils.pydantic_ndarray import NDArrayPydantic

if TYPE_CHECKING:
    from typing_extensions import Self

ScalarBoundType = Union[int, float]
BoundType = Union[
    ScalarBoundType,
    list[ScalarBoundType],
    tuple[ScalarBoundType],
    NDArrayPydantic[int],
    NDArrayPydantic[float],
]
BoundArray = Union[IntegerArray, RealArray]


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


class Variable(BaseModel, validate_assignment=True):
    """A variable.

    A variable is defined by
    a size,
    a data type
    and the bounds of its components.

    When ``size > 1``, a bound could be defined with a scalar, in that case the bound
     ill be converted to a numpy array of the expected ``size``.
    """

    size: PositiveInt = 1
    """The size of the variable."""

    type: DataType = DataType.FLOAT
    """The type of data."""

    lower_bound: BoundType = -inf
    """The lower bound of the variable."""

    upper_bound: BoundType = inf
    """The upper bound of the variable."""

    __LOWER_BOUND: Final[str] = "lower_bound"
    __UPPER_BOUND: Final[str] = "upper_bound"
    __VALIDATE_ASSIGNMENT: Final[str] = "validate_assignment"

    @model_validator(mode="after")
    def __validate_variable(self) -> Self:
        """Validate the variable.

        Returns:
            The instance.
        """
        for name in (self.__LOWER_BOUND, self.__UPPER_BOUND):
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
            return

        if isinstance(bound, Real):
            # inf cannot be cast to int and other components rely on this value.
            dtype = None if bound in (-inf, inf) else TYPE_MAP[self.type]
            bound = full(self.size, bound, dtype=dtype)
        else:
            bound = atleast_1d(bound)

        # Temporary remove assignment validation to avoid recursion when using setattr.
        self.model_config[self.__VALIDATE_ASSIGNMENT] = False
        setattr(self, bound_name, bound)
        self.model_config[self.__VALIDATE_ASSIGNMENT] = True

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
                f"{', '.join([f'{bound[i]} (index {i})' for i in indices])}."
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
                    f"{', '.join([f'{bound[i]} (index {i})' for i in indices])}."
                )
                raise ValueError(msg)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, self.__class__)
            and self.size == other.size
            and self.type == other.type
            and (self.lower_bound == other.lower_bound).all()
            and (self.upper_bound == other.upper_bound).all()
        )
