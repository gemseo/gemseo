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

from typing import TYPE_CHECKING

from gemseo.post.dataset.dataset_plot import DatasetPlot

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from gemseo.datasets.dataset import Dataset


class Boxplot(DatasetPlot):
    """Draw the boxplots of some variables from a :class:`.Dataset`."""

    # TODO: API: remove this attribute and use the option instead.
    opacity_level: float
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
        opacity_level: float = 0.25,
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
            opacity_level: The opacity level for the faces, between 0 and 1.
            **boxplot_options: The options of the wrapped boxplot function.
        """  # noqa: D205, D212, D415
        self.__n_datasets = 1 + len(datasets)
        self.__names = dataset.get_columns(variables)
        self.__origin = 0
        self.opacity_level = opacity_level
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
            opacity_level=opacity_level,
        )

    def _create_specific_data_from_dataset(
        self,
    ) -> tuple[Sequence[str], list[float], int, float, list[str], float]:
        """
        Returns:
            The names of the variables,
            the positions of the variables on the x-axis,
            the number of datasets,
            the x-offset,
            the names of the variables,
            the level of opacity.
        """  # noqa: D205, D212, D415
        self._set_color(self.__n_datasets)
        return (
            self._specific_settings.variables or self.dataset.variable_names,
            [
                (self.__n_datasets - 1) / self.__n_datasets + i * self.__n_datasets
                for i, _ in enumerate(self.__names)
            ],
            self.__n_datasets,
            self.__origin,
            self.__names,
            self._specific_settings.opacity_level,
        )
