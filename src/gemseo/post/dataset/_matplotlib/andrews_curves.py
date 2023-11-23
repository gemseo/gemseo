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
"""The Andrews curves based on matplotlib."""

from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib import pyplot as plt
from pandas.plotting import andrews_curves

from gemseo.post.dataset._matplotlib.plot import MatplotlibPlot

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


class AndrewsCurves(MatplotlibPlot):
    """The Andrews curves based on matplotlib."""

    def _create_figures(
        self, fig: Figure | None, axes: Axes | None, column: tuple[str, str, int]
    ) -> list[Figure]:
        """
        Args:
            column: The column of the dataset containing the group names.
        """  # noqa: D205 D212 D415
        fig, axes = self._get_figure_and_axes(fig, axes)
        andrews_curves(self._common_dataset, column, ax=axes)
        plt.xlabel(self._common_settings.xlabel)
        plt.ylabel(self._common_settings.ylabel)
        plt.title(self._common_settings.title)
        return [fig]
