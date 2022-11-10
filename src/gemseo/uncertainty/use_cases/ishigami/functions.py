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
"""The Ishigami function and its gradient."""
from __future__ import annotations

from numpy import array
from numpy import cos
from numpy import ndarray
from numpy import sin


def compute_output(x: ndarray) -> float:
    """Compute the output of the Ishigami function.

    Args:
        x: The input values.

    Returns:
        The output value.
    """
    return sin(x[0]) * (1 + 0.1 * x[2] ** 4) + 7 * sin(x[1]) ** 2


def compute_gradient(x: ndarray) -> ndarray:
    """Compute the gradient of the Ishigami function.

    Args:
        x: The input values.

    Returns:
        The value of the gradient of the Ishigami function.
    """
    return array(
        [
            cos(x[0]) * (1 + 0.1 * x[2] ** 4),
            14 * sin(x[1]) * cos(x[1]),
            0.4 * sin(x[0]) * x[2] ** 3,
        ]
    )
