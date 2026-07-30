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
"""Helpers shared by the backends of `Heatmap`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import nanmax
from numpy import nanmin

if TYPE_CHECKING:
    from gemseo.util.typing import RealArray


def compute_centered_bounds(
    data: RealArray, center: float
) -> tuple[float, float] | None:
    """Compute the bounds of a colormap centered on a value, if relevant.

    Args:
        data: The data, possibly containing `NaN` values.
        center: The value at which to center the colormap.

    Returns:
        The minimum and maximum values of the data
        if `center` is not `None`
        and the data takes values on both sides of `center`,
        `None` otherwise.
    """
    data_min = nanmin(data)
    data_max = nanmax(data)
    if data_min < center < data_max:
        return data_min, data_max

    return None
