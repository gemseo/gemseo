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
"""Common typing definitions."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import MutableMapping
from typing import TYPE_CHECKING
from typing import Any
from typing import TypeVar
from typing import Union

from numpy import bool_
from numpy import complexfloating
from numpy import floating
from numpy import inexact
from numpy import integer
from numpy import number
from numpy.typing import NDArray

from gemseo.utils.compatibility.scipy import SparseArrayType

if TYPE_CHECKING:
    from gemseo.core.derivatives.jacobian_operator import JacobianOperator


BooleanArray = NDArray[bool_]
"""A NumPy array of boolean numbers."""

NumberArray = NDArray[number[Any]]
"""A NumPy array of integer or real or complex numbers."""

IntegerArray = NDArray[integer[Any]]
"""A NumPy array of integer numbers."""

RealArray = NDArray[floating[Any]]
"""A NumPy array of real numbers."""

ComplexArray = NDArray[complexfloating[Any, Any]]
"""A NumPy array of complex numbers."""

RealOrComplexArray = NDArray[inexact[Any]]
"""A NumPy array of real or complex numbers."""

RealOrComplexArrayT = TypeVar("RealOrComplexArrayT", RealArray, ComplexArray)
"""A NumPy array of real or complex numbers generic type."""

SparseOrDenseRealArray = Union[RealArray, SparseArrayType]
"""A dense NumPy array or a sparse SciPy array."""

JacobianData = MutableMapping[
    str, MutableMapping[str, Union[SparseOrDenseRealArray, "JacobianOperator"]]
]
"""A Jacobian data structure of the form ``{output_name: {input_name: jacobian}}."""

StrKeyMapping = Mapping[str, Any]
"""A read-only mapping from strings to anything."""

MutableStrKeyMapping = MutableMapping[str, Any]
"""A mutable mapping from strings to anything."""
