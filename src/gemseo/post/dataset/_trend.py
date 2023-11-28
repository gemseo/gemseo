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
"""Stuff to create trend functions for scatter plots."""

from __future__ import annotations

from typing import Callable
from typing import Final

from numpy import poly1d
from numpy import polyfit
from numpy.typing import NDArray
from scipy.interpolate import Rbf
from strenum import StrEnum

TrendFunctionCreator = Callable[
    [NDArray[float], NDArray[float]], Callable[[NDArray[float]], NDArray[float]]
]


def _create_polynomial_trend_function_creator(degree: int = 1) -> TrendFunctionCreator:
    """Create a function to create polynomial trend functions.

    Args:
        degree: The degree of the polynomial.

    Returns:
        A function to create polynomial trend functions.
    """

    def create_polynomial_trend_function(
        x: NDArray[float], y: NDArray[float]
    ) -> Callable[[NDArray[float]], NDArray[float]]:
        """Create a polynomial trend function from y data depending on x data.

        Args:
            x: The x data.
            y: The y data.

        Returns:
            The polynomial trend function.
        """
        return poly1d(polyfit(x, y, degree))

    return create_polynomial_trend_function


def _create_radial_basis_function(
    x: NDArray[float], y: NDArray[float]
) -> Callable[[NDArray[float]], NDArray[float]]:
    """Create a radial basis function.

    Args:
        x: The input samples.
        y: The output samples.

    Returns:
        The radial basis function.
    """
    return Rbf(x, y)


class Trend(StrEnum):
    """A type of trend."""

    NONE = "none"
    LINEAR = "linear"
    QUADRATIC = "quadratic"
    CUBIC = "cubic"
    RBF = "rbf"


TREND_FUNCTIONS: Final[dict[Trend, TrendFunctionCreator]] = {
    Trend.LINEAR: _create_polynomial_trend_function_creator(1),
    Trend.QUADRATIC: _create_polynomial_trend_function_creator(2),
    Trend.CUBIC: _create_polynomial_trend_function_creator(3),
    Trend.RBF: _create_radial_basis_function,
}
