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
"""Class for the empirical estimation of statistics from a dataset.

Overview
--------

The :class:`.EmpiricalStatistics` class inherits
from the abstract :class:`.Statistics` class
and aims to estimate statistics from a :class:`.Dataset`,
based on empirical estimators.

Construction
------------

A :class:`.EmpiricalStatistics` is built from a :class:`.Dataset`
and optionally variables names.
In this case,
statistics are only computed for these variables.
Otherwise,
statistics are computed for all the variable available in the dataset.
Lastly,
the user can give a name to its :class:`.EmpiricalStatistics` object.
By default,
this name is the concatenation of 'EmpiricalStatistics'
and the name of the :class:`.Dataset`.
"""

from __future__ import annotations

from operator import ge
from operator import le
from typing import TYPE_CHECKING
from typing import Any
from typing import Final

from numpy import all as np_all
from numpy import cumsum
from numpy import linspace
from numpy import max as np_max
from numpy import mean
from numpy import min as np_min
from numpy import ndarray
from numpy import quantile
from numpy import std
from numpy import unique
from numpy import var
from scipy.stats import gaussian_kde
from scipy.stats import moment

from gemseo.datasets.dataset import Dataset
from gemseo.post.dataset.boxplot import Boxplot
from gemseo.post.dataset.lines import Lines
from gemseo.uncertainty.statistics.statistics import Statistics
from gemseo.utils.string_tools import repr_variable

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


class EmpiricalStatistics(Statistics):
    """A toolbox to compute statistics empirically.

    Unless otherwise stated,
    the statistics are computed *variable-wise* and *component-wise*,
    i.e. variable-by-variable and component-by-component.
    So, for the sake of readability,
    the methods named as :meth:`compute_statistic` return ``dict[str, ndarray]`` objects
    whose values are the names of the variables
    and the values are the statistic estimated for the different component.

    Examples:
        >>> from gemseo import (
        ...     create_discipline,
        ...     create_parameter_space,
        ...     create_scenario,
        ... )
        >>> from gemseo.uncertainty.statistics.empirical import EmpiricalStatistics
        >>>
        >>> expressions = {"y1": "x1+2*x2", "y2": "x1-3*x2"}
        >>> discipline = create_discipline(
        ...     "AnalyticDiscipline", expressions=expressions
        ... )
        >>>
        >>> parameter_space = create_parameter_space()
        >>> parameter_space.add_random_variable(
        ...     "x1", "OTUniformDistribution", minimum=-1, maximum=1
        ... )
        >>> parameter_space.add_random_variable(
        ...     "x2", "OTUniformDistribution", minimum=-1, maximum=1
        ... )
        >>>
        >>> scenario = create_scenario(
        ...     [discipline],
        ...     "DisciplinaryOpt",
        ...     "y1",
        ...     parameter_space,
        ...     scenario_type="DOE",
        ... )
        >>> scenario.execute({"algo": "OT_MONTE_CARLO", "n_samples": 100})
        >>>
        >>> dataset = scenario.to_dataset(opt_naming=False)
        >>>
        >>> statistics = EmpiricalStatistics(dataset)
        >>> mean = statistics.compute_mean()
    """

    __CDF_LABEL: Final[str] = "CDF"
    """The label for the cumulative distribution function."""

    __PDF_LABEL: Final[str] = "PDF"
    """The label for the probability density function."""

    def compute_maximum(self) -> dict[str, ndarray]:  # noqa: D102
        return {
            name: np_max(self.dataset.get_view(variable_names=name).to_numpy(), 0)
            for name in self.names
        }

    def compute_mean(self) -> dict[str, ndarray]:  # noqa: D102
        return {
            name: mean(self.dataset.get_view(variable_names=name).to_numpy(), 0)
            for name in self.names
        }

    def compute_minimum(self) -> dict[str, ndarray]:  # noqa: D102
        return {
            name: np_min(self.dataset.get_view(variable_names=name).to_numpy(), 0)
            for name in self.names
        }

    def compute_probability(  # noqa: D102
        self, thresh: Mapping[str, float | ndarray], greater: bool = True
    ) -> dict[str, ndarray]:
        operator = ge if greater else le
        return {
            name: mean(
                operator(
                    self.dataset.get_view(variable_names=name).to_numpy(), thresh[name]
                ),
                0,
            )
            for name in self.names
        }

    def compute_joint_probability(  # noqa: D102
        self, thresh: Mapping[str, float | ndarray], greater: bool = True
    ) -> dict[str, float]:
        operator = ge if greater else le
        return {
            name: mean(
                np_all(
                    operator(
                        self.dataset.get_view(variable_names=name).to_numpy(),
                        thresh[name],
                    ),
                    1,
                )
            )
            for name in self.names
        }

    def compute_quantile(self, prob: float) -> dict[str, ndarray]:  # noqa: D102
        return {
            name: quantile(
                self.dataset.get_view(variable_names=name).to_numpy(), prob, 0
            )
            for name in self.names
        }

    def compute_standard_deviation(self) -> dict[str, ndarray]:  # noqa: D102
        return {
            name: std(self.dataset.get_view(variable_names=name).to_numpy(), 0)
            for name in self.names
        }

    def compute_variance(self) -> dict[str, ndarray]:  # noqa: D102
        return {
            name: var(self.dataset.get_view(variable_names=name).to_numpy(), 0)
            for name in self.names
        }

    def compute_moment(self, order: int) -> dict[str, ndarray]:  # noqa: D102
        return {
            name: moment(self.dataset.get_view(variable_names=name).to_numpy(), order)
            for name in self.names
        }

    def compute_range(self) -> dict[str, ndarray]:  # noqa: D102
        lower = self.compute_minimum()
        return {
            name: upper - lower[name] for name, upper in self.compute_maximum().items()
        }

    def plot_boxplot(
        self,
        save: bool = False,
        show: bool = True,
        directory_path: str | Path = "",
        file_format: str = "png",
        **options: Any,
    ) -> dict[str, Boxplot]:
        """Visualize the data with a boxplot.

        Args:
            save: Whether to save the figures.
            show: Whether to show the figures.
            directory_path: The path to save the figures.
            file_format: The file extension.
            **options: The options of the :class:`.Boxplot` graphs.

        Returns:
            The boxplot of each variable.
        """
        plots = {
            name: Boxplot(self.dataset, variables=[name], **options)
            for name in self.names
        }
        for plot in plots.values():
            plot.execute(
                save=save,
                show=show,
                directory_path=directory_path,
                file_format=file_format,
            )
        return plots

    def plot_pdf(
        self,
        save: bool = False,
        show: bool = True,
        directory_path: str | Path = "",
        file_format: str = "png",
        **options: Any,
    ):
        """Visualize the empirical probability density function.

        Args:
            save: Whether to save the figures.
            show: Whether to show the figures.
            directory_path: The path to save the figures.
            file_format: The file extension.
            **options: The options of the :class:`.Lines` graphs.

        Returns:
            The graph of the probability density function for each variable.
        """
        plots = {}
        for name in self.names:
            size = self.dataset.variable_names_to_n_components[name]
            for index, samples in enumerate(
                self.dataset.get_view(variable_names=name).to_numpy().T
            ):
                kde = gaussian_kde(samples)
                x = linspace(samples.min(), samples.max(), 1000)
                dataset = Dataset()
                dataset.add_variable(name, x)
                dataset.add_variable(self.__PDF_LABEL, kde(x))
                plot = Lines(dataset, [self.__PDF_LABEL], name, **options)
                xlabel = repr_variable(name, index, size)
                plot.xlabel = xlabel
                plot.ylabel = "Probability density function"
                plot.execute(
                    save=save,
                    show=show,
                    directory_path=directory_path,
                    file_format=file_format,
                )
                plots[xlabel] = plot

        return plots

    def plot_cdf(
        self,
        save: bool = False,
        show: bool = True,
        directory_path: str | Path = "",
        file_format: str = "png",
        **options: Any,
    ) -> dict[str, Lines]:
        """Visualize the empirical cumulative probability function.

        Args:
            save: Whether to save the figures.
            show: Whether to show the figures.
            directory_path: The path to save the figures.
            file_format: The file extension.
            **options: The options of the :class:`.Lines` graphs.

        Returns:
            The graph of the cumulative probability function for each variable.
        """
        plots = {}
        for name in self.names:
            size = self.dataset.variable_names_to_n_components[name]
            for index, samples in enumerate(
                self.dataset.get_view(variable_names=name).to_numpy().T
            ):
                x, y = self.__evaluate_ecdf(samples)
                dataset = Dataset()
                dataset.add_variable(self.__CDF_LABEL, x)
                dataset.add_variable(name, y)
                plot = Lines(dataset, [self.__CDF_LABEL], name, **options)
                xlabel = repr_variable(name, index, size)
                plot.xlabel = xlabel
                plot.ylabel = "Cumulative probability function"
                plot.execute(
                    save=save,
                    show=show,
                    directory_path=directory_path,
                    file_format=file_format,
                )
                plots[xlabel] = plot

        return plots

    @staticmethod
    def __evaluate_ecdf(data: ndarray) -> tuple[ndarray, ndarray]:
        """Evaluate the empirical cumulative distribution function (ECDF).

        Args:
            data: The data (one sample per row).

        Returns:
            The quantiles of the random variable,
            the evaluations of the empirical cumulative distribution function (ECDF)
            for these quantiles.
        """
        quantiles, counts = unique(data, return_counts=True)
        return quantiles, cumsum(counts) / data.size
