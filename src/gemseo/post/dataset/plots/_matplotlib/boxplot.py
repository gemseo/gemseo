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
"""A boxplot based on matplotlib."""

from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib import pyplot as plt

from gemseo.post.dataset.plots._matplotlib.plot import MatplotlibPlot

if TYPE_CHECKING:
    from collections.abc import Iterable

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


class Boxplot(MatplotlibPlot):
    """A boxplot based on matplotlib."""

    def _create_figures(
        self,
        fig: Figure | None,
        ax: Axes | None,
        variable_names: Iterable[str],
        positions: Iterable[float],
        n_datasets: int,
        origin: float,
        names: Iterable[str],
        opacity_level: float,
    ) -> list[Figure]:
        """
        Args:
            variable_names: The names of the variables.
            positions: The positions of the variables on the x-axis.
            n_datasets: The number of datasets.
            origin: The x-offset.
            names: The names of the columns.
            opacity_level: The level of opacity.
        """  # noqa: D205, D212, D415
        fig, ax = self._get_figure_and_axes(fig, ax)
        self._common_settings.set_colors(self._common_settings.color)
        self.__draw_boxplot(
            self._common_dataset,
            ax,
            variable_names,
            self._common_settings.color[0],
            n_datasets,
            origin,
            names,
        )
        for index, dataset in enumerate(self._specific_settings.datasets):
            self.__draw_boxplot(
                dataset,
                ax,
                variable_names,
                self._common_settings.color[index + 1],
                n_datasets,
                origin + index + 1,
                names,
            )

        if self._specific_settings.use_vertical_bars:
            ax.set_xticks(positions)
            ax.set_xticklabels(names)
        else:
            ax.set_yticks(positions)
            ax.set_yticklabels(names)

        ax.set_xlabel(self._common_settings.xlabel)
        ax.set_ylabel(self._common_settings.ylabel)
        ax.set_title(self._common_settings.title)

        if n_datasets > 1:
            plt.plot(
                [], c=self._common_settings.color[0], label=self._common_dataset.name
            )
            for index in range(n_datasets - 1):
                plt.plot(
                    [],
                    c=self._common_settings.color[index + 1],
                    label=self._specific_settings.datasets[index].name,
                )
            plt.legend(loc="upper right")

        return [fig]

    def __draw_boxplot(self, *args) -> None:
        dataset, ax, variables, color, n_datasets, origin, names = args
        if self._specific_settings.center or self._specific_settings.scale:
            dataset = dataset.get_normalized(
                use_min_max=False,
                center=self._specific_settings.center,
                scale=self._specific_settings.scale,
            )
        boxplot = ax.boxplot(
            dataset.get_view(variable_names=variables).to_numpy(),
            vert=self._specific_settings.use_vertical_bars,
            notch=self._specific_settings.add_confidence_interval,
            showfliers=self._specific_settings.add_outliers,
            positions=[origin + i * n_datasets for i, _ in enumerate(names)],
            sym="*",
            patch_artist=True,
            flierprops={"markeredgecolor": color},
            **self._specific_settings.boxplot_options,
        )
        if self._common_settings.grid:
            ax.xaxis.grid(
                visible=True, linestyle="-", which="major", color="lightgrey", alpha=0.5
            )
            ax.yaxis.grid(
                visible=True, linestyle="-", which="major", color="lightgrey", alpha=0.5
            )

        plt.setp(boxplot["boxes"], color=color)
        plt.setp(boxplot["whiskers"], color=color)
        plt.setp(boxplot["caps"], color=color)
        plt.setp(boxplot["medians"], color=color)
        for patch in boxplot["boxes"]:
            patch.set(alpha=self._specific_settings.opacity_level)
