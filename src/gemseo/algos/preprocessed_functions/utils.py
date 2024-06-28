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
"""Utils for preprocessed functions."""

from __future__ import annotations

from numpy import isnan
from numpy import ndarray

from gemseo.algos.stop_criteria import DesvarIsNan
from gemseo.algos.stop_criteria import FunctionIsNan


def check_function_output_includes_nan(
    value: ndarray,
    stop_if_nan: bool = True,
    function_name: str = "",
    xu_vect: ndarray | None = None,
) -> None:
    """Check if an array contains a NaN value.

    Args:
        value: The array to be checked.
        stop_if_nan: Whether to stop if `value` contains a NaN.
        function_name: The name of the function.
            If empty,
            the arguments ``function_name`` and ``xu_vect`` are ignored.
        xu_vect: The point at which the function is evaluated.
            ``None`` if and only if ``function_name`` is empty.

    Raises:
        DesvarIsNan: If the value is a function input containing a NaN.
        FunctionIsNan: If the value is a function output containing a NaN.
    """
    if stop_if_nan and isnan(value).any():
        if function_name:
            msg = (
                f"The function {function_name} contains a NaN value "
                f"for x={xu_vect}."
            )
            raise FunctionIsNan(msg)

        msg = f"The input vector contains a NaN value: {value}."
        raise DesvarIsNan(msg)
