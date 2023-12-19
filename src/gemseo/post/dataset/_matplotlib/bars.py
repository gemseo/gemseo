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
"""A bar plot based on matplotlib."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import arange

from gemseo.post.dataset._matplotlib.plot import MatplotlibPlot

if TYPE_CHECKING:
    from collections.abc import Iterable

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray


class BarPlot(MatplotlibPlot):
    """A bar plot based on matplotlib."""

    def _create_figures(
        self,
        fig: Figure | None,
        axes: Axes | None,
        data: NDArray,
        feature_names: Iterable[str],
    ) -> list[Figure]:
        """
        Args:
            data: The data to be plotted.
            feature_names: The names of the features.
        """  # noqa: D205, D212, D415
        fig, axes = self._get_figure_and_axes(fig, axes)
        n_series, n_features = data.shape
        axes.tick_params(labelsize=self._common_settings.font_size)
        first_series_positions = arange(n_features)
        width = 0.75 / n_series
        subplots = []
        positions = [
            first_series_positions + index * width + width / 2
            for index in range(n_series)
        ]
        for feature_positions, series_name, series_data, series_color in zip(
            positions,
            self._common_dataset.index,
            data,
            self._common_settings.color,
        ):
            subplots.append(
                axes.bar(
                    feature_positions,
                    series_data.tolist(),
                    width,
                    label=series_name,
                    color=series_color,
                )
            )

        for rects in subplots:
            for rect in rects:
                height = rect.get_height()
                pos = 3 if height > 0 else -12
                axes.annotate(
                    f"{round(height, self._specific_settings.n_digits)}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, pos),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                )

        axes.set_xticks([position + 0.375 for position in first_series_positions])
        axes.set_xticklabels(feature_names)
        axes.set_xlabel(self._common_settings.xlabel)
        axes.set_ylabel(self._common_settings.ylabel)
        axes.set_title(
            self._common_settings.title, fontsize=self._common_settings.font_size * 1.2
        )
        axes.legend(fontsize=self._common_settings.font_size)
        axes.set_axisbelow(True)
        axes.grid()
        return [fig]
