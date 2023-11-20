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

import matplotlib
import matplotlib.pyplot as plt
from packaging import version

if version.parse(matplotlib.__version__) < version.parse("3.5.0"):

    def get_color_map(colormap):  # noqa: N802, D103
        return plt.cm.get_cmap(colormap)

else:

    def get_color_map(colormap):  # noqa: N802, D103
        return plt.colormaps[colormap]
