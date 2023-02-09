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
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Draw the boxplots of some variables from a :class:`.Dataset`.

A boxplot represents the median and the first and third quartiles of numerical data. The
variability outside the inter quartile domain can be represented with lines, called
*whiskers*. The numerical data that are significantly different are called *outliers*
and can be plotted as individual points beyond the whiskers.
"""
from __future__ import annotations

from typing import Any
from typing import ClassVar
from typing import Iterable
from typing import Sequence

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from gemseo.core.dataset import Dataset
from gemseo.post.dataset.dataset_plot import DatasetPlot


class Boxplot(DatasetPlot):
    """Draw the boxplots of some variables from a :class:`.Dataset`."""

    opacity_level: ClassVar[float] = 0.25
    """The opacity level for the faces, between 0 and 1."""

    def __init__(
        self,
        dataset: Dataset,
        *datasets: Dataset,
        variables: Sequence[str] | None = None,
        center: bool = False,
        scale: bool = False,
        use_vertical_bars: bool = True,
        add_confidence_interval: bool = False,
        add_outliers: bool = True,
        **boxplot_options: Any,
    ) -> None:
        """
        Args:
            *datasets: Datasets containing other series of data to plot.
            variables: The names of the variables to plot.
                If ``None``, use all the variables.
            center: Whether to center the variables so that they have a zero mean.
            scale: Whether to scale the variables so that they have a unit variance.
            use_vertical_bars: Whether to use vertical bars.
            add_confidence_interval: Whether to add the confidence interval (CI)
                around the median; a CI is also called *notch*.
            add_outliers: Whether to add the outliers.
            **boxplot_options: The options of the wrapped boxplot function.
        """  # noqa: D205, D212, D415
        super().__init__(
            dataset,
            datasets=datasets,
            variables=variables,
            center=center,
            scale=scale,
            use_vertical_bars=use_vertical_bars,
            add_confidence_interval=add_confidence_interval,
            add_outliers=add_outliers,
            boxplot_options=boxplot_options,
        )
        self.__n_datasets = 1 + len(datasets)
        self.__names = self.dataset.get_column_names(variables)
        self.__origin = 0

    def _plot(
        self,
        fig: None | Figure = None,
        axes: None | Axes = None,
    ) -> list[Figure]:
        fig, axes = self._get_figure_and_axes(fig, axes)
        variables = self._param.variables
        if variables is None:
            variables = self.dataset.variables

        self._set_color(self.__n_datasets)
        self.__draw_boxplot(self.dataset, axes, variables, self.color[0])
        for index, dataset in enumerate(self._param.datasets):
            self.__draw_boxplot(dataset, axes, variables, self.color[index + 1])

        positions = [
            (self.__n_datasets - 1) / self.__n_datasets + i * self.__n_datasets
            for i, _ in enumerate(self.__names)
        ]
        if self._param.use_vertical_bars:
            axes.set_xticks(positions)
            axes.set_xticklabels(self.__names)
        else:
            axes.set_yticks(positions)
            axes.set_yticklabels(self.__names)

        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)
        axes.set_title(self.title)

        if self.__n_datasets > 1:
            plt.plot([], c=self.color[0], label=self.dataset.name)
            for index in range(self.__n_datasets - 1):
                plt.plot(
                    [], c=self.color[index + 1], label=self._param.datasets[index].name
                )
            plt.legend(loc="upper right")

        return [fig]

    def __draw_boxplot(
        self, dataset: Dataset, axes: Axes, variables: Iterable[str], color: str
    ) -> None:
        """Draw the boxplots for a given dataset.

        Args:
            dataset: The dataset containing the data to be plotted.
            axes: The axes to plot the data.
            variables: The names of the variables.
            color: The color for the boxplot.
        """
        if self._param.center or self._param.scale:
            dataset = dataset.get_normalized_dataset(
                use_min_max=False, center=self._param.center, scale=self._param.scale
            )
        boxplot = axes.boxplot(
            dataset.get_data_by_names(variables, False),
            vert=self._param.use_vertical_bars,
            notch=self._param.add_confidence_interval,
            showfliers=self._param.add_outliers,
            positions=[
                self.__origin + i * self.__n_datasets
                for i, _ in enumerate(self.__names)
            ],
            sym="*",
            patch_artist=True,
            flierprops=dict(markeredgecolor=color),
            **self._param.boxplot_options,
        )
        self.__origin += 1

        axes.xaxis.grid(
            True, linestyle="-", which="major", color="lightgrey", alpha=0.5
        )
        axes.yaxis.grid(
            True, linestyle="-", which="major", color="lightgrey", alpha=0.5
        )

        plt.setp(boxplot["boxes"], color=color)
        plt.setp(boxplot["whiskers"], color=color)
        plt.setp(boxplot["caps"], color=color)
        plt.setp(boxplot["medians"], color=color)
        for patch in boxplot["boxes"]:
            patch.set(alpha=self.opacity_level)
