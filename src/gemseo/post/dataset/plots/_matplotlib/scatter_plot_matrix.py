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
"""Scatter matrix based on matplotlib."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from matplotlib import pyplot as plt
from matplotlib.pyplot import colormaps
from numpy import ptp
from pandas.plotting import scatter_matrix

from gemseo.post.dataset._trend import TREND_FUNCTIONS
from gemseo.post.dataset._trend import Trend
from gemseo.post.dataset.plots._matplotlib.plot import MatplotlibPlot
from gemseo.post.dataset.scatter_plot_matrix_settings import ScatterMatrix_Settings

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


class ScatterMatrix(MatplotlibPlot[ScatterMatrix_Settings]):
    """Scatter matrix based on matplotlib."""

    def _create_figures(
        self,
        fig: Figure | None,
        ax: Axes | None,
        classifier_column: tuple[str, str, int],
    ) -> list[Figure]:
        """
        Args:
            classifier_column: The column of the dataset used for classification.
        """  # noqa: D212, D205
        settings = self._settings
        variable_names = settings.variable_names
        classifier = settings.classifier
        kde = settings.kde
        size = settings.size
        marker = settings.marker
        dataframe = self._common_dataset.get_view(variable_names=variable_names)
        kwargs = {}
        if classifier:
            values = self._common_dataset.get_view(
                variable_names=[classifier]
            ).to_numpy()[:, 0]
            values = (values - values.min()) / ptp(values)
            colormap = colormaps[settings.colormap_name]
            kwargs["color"] = [colormap(value) for value in values]
            if settings.exclude_classifier:
                dataframe = dataframe.drop(labels=classifier_column, axis=1)

        dataframe.columns = self._get_variable_names(dataframe)
        n_cols = n_rows = dataframe.shape[1] if ax is None else 1
        fig, ax = self._get_figure_and_axes(
            fig,
            ax,
            fig_size=settings.fig_size,
            n_rows=n_rows,
            n_cols=n_cols,
        )
        axs = scatter_matrix(
            dataframe,
            diagonal="kde" if kde else "hist",
            s=size,
            marker=marker,
            figsize=settings.fig_size,
            ax=ax,
            # The grid argument is ignored because the subplots do not have axes.
            # See the issue https://github.com/pandas-dev/pandas/issues/50818.
            grid=settings.grid,
            **kwargs,
            **settings.options,
        )

        trend_function_creator = settings.trend
        if trend_function_creator != Trend.NONE:
            if not isinstance(trend_function_creator, Callable):
                trend_function_creator = TREND_FUNCTIONS[trend_function_creator]

            for i_row, row in enumerate(axs):
                for i_col, ax in enumerate(row):
                    if i_col == i_row:
                        continue

                    collection = ax.collections[0]
                    collection.set_zorder(3)
                    data = collection.get_offsets()
                    data = data[data[:, 0].argsort()]
                    x_data = data[:, 0]
                    trend_function = trend_function_creator(x_data, data[:, 1])
                    ax.plot(
                        x_data,
                        trend_function(x_data),
                        color="gray",
                        linestyle="--",
                    )

        n_cols = axs.shape[0]
        if not (settings.plot_lower and settings.plot_upper):
            for i in range(n_cols):
                for j in range(n_cols):
                    axs[i, j].get_xaxis().set_visible(False)
                    axs[i, j].get_yaxis().set_visible(False)

        if not settings.plot_lower:
            for i in range(n_cols):
                for j in range(i):
                    axs[i, j].set_visible(False)

            for i in range(n_cols):
                axs[i, i].get_xaxis().set_visible(True)
                axs[i, i].get_yaxis().set_visible(True)

        if not settings.plot_upper:
            for i in range(n_cols):
                for j in range(i + 1, n_cols):
                    axs[i, j].set_visible(False)

            for i in range(n_cols):
                axs[-1, i].get_xaxis().set_visible(True)
                axs[i, 0].get_yaxis().set_visible(True)

        plt.suptitle(settings.title)
        return [fig]
