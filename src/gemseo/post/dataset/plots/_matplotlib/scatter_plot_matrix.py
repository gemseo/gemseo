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
"""Pair plot based on matplotlib."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from matplotlib import pyplot as plt
from matplotlib.pyplot import colormaps
from numpy import linspace
from numpy import mgrid
from numpy import unique
from numpy import vstack
from scipy.stats import gaussian_kde

from gemseo.post.dataset.pair_plot_settings import PairPlot_Settings
from gemseo.post.dataset.plots._matplotlib.plot import MatplotlibPlot
from gemseo.post.dataset.trend import _TREND_FUNCTIONS
from gemseo.post.dataset.trend import Trend

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy import ndarray

    from gemseo.typing import RealArray
    from gemseo.typing import StrKeyMapping


class PairPlot(MatplotlibPlot[PairPlot_Settings]):
    """Pair plot based on matplotlib."""

    def _create_figures(
        self,
        fig: Figure | None,
        ax: Axes | None,
        classifier_column: tuple[str, str, int],
    ) -> list[Figure]:
        """
        Args:
            classifier_column: The column of the dataset used for classification.

        Raises:
            ValueError: If the arguments `fig` and `ax` are not ``None``.
        """  # noqa: D212, D205
        if fig is not None or ax is not None:
            msg = "The arguments 'fig' and 'ax' are not supported by PairPlot."
            raise ValueError(msg)

        settings = self._settings
        classifier = settings.classifier
        use_kde = settings.use_kde
        use_ranks = settings.use_ranks
        size = settings.size
        marker = settings.marker
        dataset = self._common_dataset
        if settings.use_scatter:
            options = {"alpha": 0.5}
        else:
            options = {"levels": 10, "cmap": settings.colormap_name}

        options.update(settings.options)

        # Create categories (i.e. masks and colors) if classifier.
        if classifier:
            cat_values = dataset.get_view(variable_names=[classifier]).to_numpy()[:, 0]
            categories = unique(cat_values)
            n_categories = len(categories)
            colormap = colormaps[settings.colormap_name]
            colors = [
                colormap(i / max(n_categories - 1, 1)) for i in range(n_categories)
            ]
            masks = [cat_values == cat for cat in categories]
        else:
            masks = []
            colors = []

        # Get raw data.
        dataframe = dataset.get_view(variable_names=settings.variable_names)
        if classifier and settings.exclude_classifier:
            dataframe = dataframe.drop(labels=classifier_column, axis=1)
        dataframe.columns = self._get_variable_names(dataframe)
        columns = list(dataframe.columns)
        n = len(columns)
        data = dataframe.to_numpy()

        # Set upper data (ranked data if required, otherwise raw data).
        upper_data = dataframe.rank(pct=True).to_numpy() if use_ranks else data

        # Create empty figure.
        fig, axs = plt.subplots(n, n, figsize=settings.fig_size)

        # Draw diagonal blocks, using either histogram or 1-D KDE.
        # If classifier, add one layer per category.
        for i in range(n):
            ax = axs[i, i]
            data_i = data[:, i]
            if classifier:
                if use_kde:
                    for mask, color in zip(masks, colors, strict=True):
                        self.__plot_1d_kde(ax, data_i, mask, color)
                else:
                    for mask, color in zip(masks, colors, strict=True):
                        ax.hist(data_i[mask], color=color, alpha=0.5)
            elif use_kde:
                self.__plot_1d_kde(ax, data_i)
            else:
                ax.hist(data_i)

            self.__set_metadata(ax, columns, i, i)

        # Draw off-diagonal blocks, using either scatter plot or 2-D KDE.
        # If classifier, add one layer per category.
        for i, row in enumerate(axs):
            for j, ax in enumerate(row):
                if i == j:
                    continue

                if i < j:
                    #
                    data_i = upper_data[:, i]
                    data_j = upper_data[:, j]
                else:
                    data_i = data[:, i]
                    data_j = data[:, j]

                if settings.use_scatter:
                    if classifier:
                        for mask, color in zip(masks, colors, strict=True):
                            ax.scatter(
                                data_j[mask],
                                data_i[mask],
                                s=size,
                                marker=marker,
                                color=color,
                                **options,
                            )
                    else:
                        ax.scatter(
                            data_j,
                            data_i,
                            s=size,
                            marker=marker,
                            **options,
                        )
                else:
                    x_min = data_j.min()
                    x_max = data_j.max()
                    y_min = data_i.min()
                    y_max = data_i.max()
                    xx, yy = mgrid[x_min:x_max:100j, y_min:y_max:100j]
                    positions = vstack([xx.ravel(), yy.ravel()])
                    if classifier:
                        options.pop("cmap", None)
                        for mask, color in zip(masks, colors, strict=True):
                            options["colors"] = color
                            self.__plot_2d_kde(
                                ax,
                                data_i[mask],
                                data_j[mask],
                                positions,
                                xx,
                                yy,
                                options,
                            )
                    else:
                        self.__plot_2d_kde(
                            ax, data_i, data_j, positions, xx, yy, options
                        )

                if use_ranks and i < j:
                    ax.set_xlim(-0.05, 1.05)
                    ax.set_ylim(-0.05, 1.05)

                self.__set_metadata(ax, columns, i, j)

        self.__add_trend(axs)
        self.__hide_sub_plots(axs)
        plt.suptitle(settings.title)
        return [fig]

    @staticmethod
    def __plot_2d_kde(
        ax: Axes,
        samples_i: RealArray,
        samples_j: RealArray,
        positions: RealArray,
        xx: RealArray,
        yy: RealArray,
        options: StrKeyMapping,
    ) -> None:
        """Plot a 2-D KDE.

        Args:
            ax: The axes to plot on.
            samples_i: The sample data for the i-th variable.
            samples_j: The sample data for the j-th variable.
            positions: The positions for evaluating the kernel density estimate.
            xx: The x-coordinates of the grid for contour plotting.
            yy: The y-coordinates of the grid for contour plotting.
            options: Options to pass to the contour function.
        """
        kde = gaussian_kde(vstack([samples_j, samples_i]))
        density = kde(positions).reshape(xx.shape)
        ax.contour(xx, yy, density, **options)

    def __set_metadata(self, ax: Axes, columns: Sequence[str], i: int, j: int) -> None:
        """Set metadata.

        Args:
            ax: The axes to plot on.
            columns: The columns of the dataset.
            i: The row index.
            j: The column index.
        """
        n = len(columns)

        ax.set_box_aspect(1)

        if self._settings.grid:
            ax.grid(True, zorder=0)
            ax.set_axisbelow(True)

        if j == 0:
            ax.set_ylabel(columns[i])

        if i == n - 1:
            ax.set_xlabel(columns[j])

        if i == j and not self._settings.plot_lower:
            ax.set_xlabel(columns[j])
            ax.set_ylabel(columns[j])

        if j == n - 1 and i < n - 1:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")

        if i == 0 and j > 0:
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position("top")

        if 0 < j < n - 1 or (i == 0 and j == 0) or (i == n - 1 and j == n - 1):
            ax.yaxis.set_ticklabels([])

        if 0 < i < n - 1 or (i == 0 and j == 0):
            ax.xaxis.set_ticklabels([])

    @staticmethod
    def __plot_1d_kde(
        ax: Axes,
        values: RealArray,
        mask: RealArray | Ellipsis = Ellipsis,
        color: tuple[float, float, float, float] | None = None,
    ) -> None:
        """Plot a 1-D KDE.

        Args:
            ax: The axes to plot on.
            values: The entire dataset.
            mask: The mask for the data of interest.
            color: The RGBA color, if any.
        """
        x_range = linspace(values.min(), values.max(), 200)
        estimator = gaussian_kde(values[mask])
        line = ax.plot(x_range, estimator(x_range))[0]
        if color is not None:
            line.set_color(color)

    def __add_trend(self, axs: ndarray) -> None:
        """Add a trend on the off-diagonal sub-plots, if required.

        Args:
            axs: The axes of the sub-plot.
        """
        trend_function_creator = self._settings.trend
        if trend_function_creator != Trend.NONE:
            if not isinstance(trend_function_creator, Callable):
                trend_function_creator = _TREND_FUNCTIONS[trend_function_creator]

            for i, row in enumerate(axs):
                for j, ax in enumerate(row):
                    if j == i:
                        continue

                    for collection in ax.collections:
                        collection.set_zorder(3)

                    offsets = vstack([c.get_offsets() for c in ax.collections])
                    offsets = offsets[offsets[:, 0].argsort()]
                    x_values = offsets[:, 0]
                    y_values = offsets[:, 1]
                    trend_function = trend_function_creator(x_values, y_values)
                    ax.plot(
                        x_values,
                        trend_function(x_values),
                        color="gray",
                        linestyle="--",
                    )

    def __hide_sub_plots(self, axs: ndarray) -> None:
        """Hide sub-plots, if required.

        Args:
            axs: The axes of the sub-plot.
        """
        if not self._settings.plot_lower:
            for i, row in enumerate(axs):
                ax = row[i]
                ax.get_xaxis().set_visible(True)
                ax.get_yaxis().set_visible(True)
                for j in range(i):
                    row[j].set_visible(False)

        if not self._settings.plot_upper:
            n = len(axs)
            for i, row in enumerate(axs):
                axs[-1, i].get_xaxis().set_visible(True)
                row[i].get_yaxis().set_visible(True)
                for j in range(i + 1, n):
                    row[j].set_visible(False)
