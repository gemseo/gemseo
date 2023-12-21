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
"""A base plot class relying on matplotlib."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple

from matplotlib import pyplot as plt

from gemseo.post.dataset.base_plot import BasePlot
from gemseo.utils.matplotlib_figure import save_show_figure

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from gemseo import FigSizeType
    from gemseo.datasets.dataset import Dataset
    from gemseo.post.dataset.plot_settings import PlotSettings


class MatplotlibPlot(BasePlot):
    """A base plot class relying on matplotlib."""

    __figures: list[Figure]
    """The matplotlib figures."""

    def __init__(
        self,
        dataset: Dataset,
        common_settings: PlotSettings,
        specific_settings: NamedTuple,
        *specific_data: Any,
        fig: Figure | None = None,
        axes: Axes | None = None,
    ) -> None:
        """
        Args:
            fig: The figure.
                If ``None``, create a new one.
            axes: The axes.
                If ``None``, create new ones.
        """  # noqa: D205 D212 D415
        super().__init__(
            dataset, common_settings, specific_settings, fig=fig, axes=axes
        )
        self.__figures = self._create_figures(fig, axes, *specific_data)
        xtick_rotation = self._common_settings.xtick_rotation
        if xtick_rotation:
            for figure in self.__figures:
                for ax in figure.axes:
                    ax.tick_params(axis="x", labelrotation=xtick_rotation)

    @abstractmethod
    def _create_figures(
        self, fig: Figure | None, axes: Axes | None, *specific_data: Any
    ) -> list[Figure]:
        """Create the matplotlib figures.

        Args:
            fig: The figure.
                If ``None``, create a new one.
            axes: The axes.
                If ``None``, create new ones.
            *specific_data: The data specific to this plot class.

        Returns:
            The matplotlib figures.
        """

    def show(self) -> None:  # noqa: D102
        for sub_figure in self.__figures:
            save_show_figure(sub_figure, True, "")

    def _save(self, file_path: Path) -> tuple[str]:
        file_paths = []
        for index, sub_figure in enumerate(self.__figures):
            if len(self.__figures) > 1:
                fig_file_path = self._file_path_manager.add_suffix(file_path, index)
            else:
                fig_file_path = file_path

            file_paths.append(str(fig_file_path))

            save_show_figure(sub_figure, False, fig_file_path)

        return tuple(file_paths)

    def _get_figure_and_axes(
        self,
        fig: Figure | None,
        axes: Axes | None,
        fig_size: FigSizeType | None = None,
        n_rows: int = 1,
        n_cols: int = 1,
    ) -> tuple[Figure, Axes]:
        """Return the figure and axes to plot the data.

        Args:
            fig: The figure to plot the data.
                If ``None``, create a new one.
            axes: The axes to plot the data.
                If ``None``, create new ones.
            fig_size: The width and height of the figure in inches.
                If ``None``, use the default ``fig_size``.
            n_rows: The number of rows of the subplot grid.
            n_cols: The number of cols of the subplot grid.

        Returns:
            The figure and axis to plot the data.
        """
        if fig is None:
            if axes is not None:
                raise ValueError(
                    "The figure associated with the given axes is missing."
                )

            return plt.subplots(
                nrows=n_rows,
                ncols=n_cols,
                figsize=fig_size or self._common_settings.fig_size,
            )

        if axes is None:
            raise ValueError("The axes associated with the given figure are missing.")

        return fig, axes

    @property
    def figures(self) -> list[Figure]:  # noqa: D102
        return self.__figures
