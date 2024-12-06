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
"""Pydantic compatible NumPy array."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import TypeVar
from typing import cast

from numpy import dtype
from numpy import ndarray

try:
    from numpy._typing._array_like import _DType_co
    from numpy._typing._array_like import _ScalarType_co
except (ModuleNotFoundError, ImportError):  # pragma: no cover
    from numpy import generic

    _DType_co = TypeVar("_DType_co", covariant=True, bound=dtype[Any])  # type: ignore[misc] # mypy seems to ignore the except block.
    _ScalarType_co = TypeVar("_ScalarType_co", bound=generic, covariant=True)  # type: ignore[misc] # mypy seems to ignore the except block.

from pydantic_core import CoreSchema
from pydantic_core import core_schema
from typing_extensions import get_args

# This type is defined in a .pyi file of NumPy,
# it cannot be imported, so it is defined here.
_ShapeType = TypeVar("_ShapeType", bound=Any)

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pydantic import GetCoreSchemaHandler


class _NDArrayPydantic(ndarray[_ShapeType, _DType_co]):
    """A ndarray that can be used with Pydantic.

    See Pydantic docs for more information.
    """

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> CoreSchema:  # noqa: D102
        if source_type is cls:
            return core_schema.is_instance_schema(ndarray)

        dtype_ = get_args(get_args(source_type)[1])[0]
        if dtype_ is _ScalarType_co:
            # No dtype has been defined.
            return core_schema.is_instance_schema(ndarray)

        return core_schema.chain_schema([
            core_schema.is_instance_schema(ndarray),
            core_schema.no_info_plain_validator_function(
                cls.__get_validator(source_type)
            ),
        ])

    @staticmethod
    def __get_validator(
        source_type: Any,
    ) -> Callable[[NDArray[Any]], NDArray[Any]]:
        """Return a function that can validate NumPy array types.

        Args:
            _source_type: The source type.

        Returns:
            The validator function.
        """
        # The dtype is located at X in ndarray[Any, dtype[X]]
        dtype_ = get_args(get_args(source_type)[1])[0]

        def validate(data: Any) -> NDArray[Any]:
            """Validate a NumPy array.

            Args:
                data: The data to validate.

            Returns:
                The data.

            Raises:
                ValidationError: If the data is not valid.
            """
            # if shape and shape is not Any and array.shape != shape:
            #     msg = f"Input shape should be {shape}: got the shape {array.shape}"
            #     raise ValueError(msg)

            # First check that the source dtype is not catch-all then the actual dtype.
            if dtype_ not in {Any, _ScalarType_co} and data.dtype != dtype_:
                msg = (
                    f"The expected dtype is {dtype_}: "
                    f"the actual dtype is {data.dtype.type}"
                )
                raise ValueError(msg)

            return cast("NDArray[Any]", data)

        return validate


NDArrayPydantic = _NDArrayPydantic[Any, dtype[_ScalarType_co]]
