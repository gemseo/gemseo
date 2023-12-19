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
"""Evolution of the variables by means of a color scale, using matplotlib."""

from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib.colors import SymLogNorm
from matplotlib.ticker import LogFormatterSciNotation
from numpy import arange
from numpy import e

from gemseo.post.dataset._matplotlib.plot import MatplotlibPlot

if TYPE_CHECKING:
    from collections.abc import Iterable

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike


class ColorEvolution(MatplotlibPlot):
    """Evolution of the variables by means of a color scale, using matplotlib."""

    def _create_figures(
        self,
        fig: Figure | None,
        axes: Axes | None,
        data: ArrayLike,
        variable_names: Iterable[str],
    ) -> list[Figure]:
        """
        Args:
            data: The data to be plotted.
            variable_names: The names of the variables.
        """  # noqa: D205, D212, D415
        fig, axes = self._get_figure_and_axes(fig, axes)
        norm = None
        if self._specific_settings.use_log:
            maximum = abs(data).max()
            norm = SymLogNorm(vmin=-maximum, vmax=maximum, linthresh=1.0, base=e)

        img_ = axes.imshow(
            data,
            cmap=self._common_settings.colormap,
            norm=norm,
            alpha=self._specific_settings.opacity,
            **self._specific_settings.options,
        )
        names = self._common_dataset.get_columns(variable_names)
        axes.set_yticks(arange(len(names)))
        axes.set_yticklabels(names)
        axes.set_xlabel(self._common_settings.xlabel)
        axes.set_ylabel(self._common_settings.ylabel)
        axes.set_title(self._common_settings.title)
        fig.colorbar(
            img_,
            ax=axes,
            format=LogFormatterSciNotation()
            if self._specific_settings.use_log
            else None,
        )
        return [fig]
