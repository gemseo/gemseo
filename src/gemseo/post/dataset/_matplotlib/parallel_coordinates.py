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
"""Parallel coordinates based on matplotlib."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import inf
from pandas.plotting import parallel_coordinates

from gemseo.post.dataset._matplotlib.plot import MatplotlibPlot

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from gemseo.datasets.dataset import Dataset


class ParallelCoordinates(MatplotlibPlot):
    """Parallel coordinates based on matplotlib."""

    def _create_figures(
        self,
        fig: Figure | None,
        axes: Axes | None,
        dataframe: Dataset,
        cluster: tuple[str, str, int],
    ) -> list[Figure]:
        """
        Args:
            dataframe: The dataset to be used.
            cluster: The identifier of the cluster.
        """  # noqa: D205, D212, D415
        fig, axes = self._get_figure_and_axes(fig, axes)
        columns = self._common_dataset.get_columns(as_tuple=True)
        axes = parallel_coordinates(
            dataframe, cluster, cols=columns, ax=axes, **self._specific_settings.kwargs
        )
        if (
            self._specific_settings.lower == -inf
            and self._specific_settings.upper == inf
        ):
            axes.get_legend().remove()
        axes.set_xticklabels(self._get_variable_names(columns))
        axes.set_xlabel(self._common_settings.xlabel)
        axes.set_ylabel(self._common_settings.ylabel)
        axes.set_title(self._common_settings.title)
        return [fig]
