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

from collections.abc import Mapping
from typing import TYPE_CHECKING
from typing import Any

from numpy import complexfloating
from numpy import floating
from numpy import inexact
from numpy import integer
from numpy import number
from numpy.typing import NDArray
from typing_extensions import Protocol
from typing_extensions import Self

NumberArray = NDArray[number]
"""A NumPy array of numbers."""

IntegerArray = NDArray[integer]
"""A NumPy array of integer numbers."""

RealArray = NDArray[floating]
"""A NumPy array of real numbers."""

ComplexArray = NDArray[complexfloating]
"""A NumPy array of complex numbers."""

RealOrComplexArray = NDArray[inexact]
"""A NumPy array of complex or real numbers."""

JacobianData = dict[str, Mapping[str, RealOrComplexArray]]
"""A Jacobian data structure of the form ``{output_name: {input_name: jacobian}}."""


if TYPE_CHECKING:

    class DataMapping(Mapping[str, Any], Protocol):
        """Common typing for dict and DisciplineData."""

        def copy(self) -> Self:
            """Return a shallow copy."""

else:

    class DataMapping(Mapping[str, Any]):
        """Common typing for dict and DisciplineData."""

        def copy(self) -> Self:
            """Return a shallow copy."""
