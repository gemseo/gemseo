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
"""Utilities shared by the tests of the dataset plots."""

from __future__ import annotations

from typing import TYPE_CHECKING

from scipy.interpolate import RBFInterpolator

if TYPE_CHECKING:
    from collections.abc import Callable

    from gemseo.util.typing import RealArray


def custom_trend(x: RealArray, y: RealArray) -> Callable[[RealArray], RealArray]:
    """Create a custom RBF trend function.

    Args:
        x: The input samples.
        y: The output samples.

    Returns:
        The trend function.
    """
    interpolator = RBFInterpolator(x.reshape(-1, 1), y)
    return lambda x_new: interpolator(x_new.reshape(-1, 1))
