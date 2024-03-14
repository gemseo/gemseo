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
"""Compatibility between different versions of matplotlib."""

from __future__ import annotations

from importlib.metadata import version
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from packaging.version import parse as parse_version

if TYPE_CHECKING:
    from matplotlib.colors import Colormap
    from matplotlib.colors import ListedColormap

if parse_version(version("matplotlib")) < parse_version("3.5.0"):

    def get_color_map(colormap: Colormap | str | None) -> ListedColormap:  # noqa: D103
        return plt.cm.get_cmap(colormap)

else:

    def get_color_map(colormap: str) -> ListedColormap:  # noqa: D103
        return plt.colormaps[colormap]
