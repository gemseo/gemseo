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
#
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial documentation
#        :author: Vincent DROUET
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Function component extractor."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import atleast_1d

if TYPE_CHECKING:
    from gemseo.core.mdo_functions.mdo_function import MDOFunction
    from gemseo.typing import NumberArray


class FunctionComponentExtractor:
    """A function to evaluate only one output component of a function."""

    __f: MDOFunction
    """The function from which to extract the component."""

    __i: int
    """The index of the component to extract."""

    def __init__(self, f: MDOFunction, i: int) -> None:
        """
        Args:
            f: The function from which to extract the component.
            i: The index of the component to extract.
        """  # noqa: D205, D212, D415
        self.__f = f
        self.__i = i

    def compute_output(self, x: NumberArray) -> float:
        """Compute the i-th output component of the function.

        Args:
            x: The input value of the function.

        Returns:
            The i-th output component of the function.
        """
        return self.__f.evaluate(x)[self.__i]

    def compute_jacobian(self, x: NumberArray) -> NumberArray:
        """Compute the Jacobian of the i-th output component of the function.

        Args:
            x: The input value of the function.

        Returns:
            The Jacobian of the i-th output component of the function.
        """
        return atleast_1d(self.__f.jac(x)[self.__i, :])
