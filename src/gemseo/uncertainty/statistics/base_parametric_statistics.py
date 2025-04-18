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
"""Parametric estimation of statistics from a dataset.

The base class :class:`.BaseParametricStatistics` aims
to estimate statistics parametrically,
using probability distributions fitted to a :class:`Dataset` at instantiation

For each variable of this :class:`Dataset`,

1. the parameters of the distributions are calibrated from this :class:`.Dataset`,
2. the fitted parametric distribution which is optimal
   in the sense of a goodness-of-fit criterion and a selection criterion
   is selected to estimate the statistics associated with this variable.

Its subclass :class:`.OTParametricStatistics` uses the OpenTURNS distributions
through the :class:`.OTDistribution` and :class:`.OTDistributionFitter` classes
and
its subclass :class:`.SPParametricStatistics` uses the SciPy distributions
through the :class:`.SPDistribution` and :class:`.SPDistributionFitter` classes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Generic
from typing import NamedTuple
from typing import TypeVar
from typing import Union

from matplotlib import pyplot as plt
from numpy import array
from numpy import linspace
from prettytable import PrettyTable
from strenum import StrEnum

from gemseo.uncertainty.distributions.base_distribution_fitter import (
    BaseDistributionFitter,
)
from gemseo.uncertainty.distributions.base_distribution_fitter import MeasureType
from gemseo.uncertainty.statistics.base_statistics import BaseStatistics
from gemseo.uncertainty.statistics.tolerance_interval.distribution import (
    BaseToleranceInterval,
)
from gemseo.uncertainty.statistics.tolerance_interval.factory import (
    ToleranceIntervalFactory,
)
from gemseo.utils.matplotlib_figure import save_show_figure
from gemseo.utils.string_tools import pretty_str
from gemseo.utils.string_tools import repr_variable

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping
    from collections.abc import Sequence

    from matplotlib.figure import Figure

    from gemseo.datasets.dataset import Dataset
    from gemseo.typing import RealArray
    from gemseo.uncertainty.distributions.base_distribution import BaseDistribution
    from gemseo.utils.matplotlib_figure import FigSizeType


_DefaultFittingCriterionT = TypeVar("_DefaultFittingCriterionT", bound=StrEnum)
_DistributionNameT = TypeVar("_DistributionNameT", bound=StrEnum)
_FittingCriterionT = TypeVar("_FittingCriterionT", bound=StrEnum)
_SignificanceTestT = TypeVar("_SignificanceTestT", bound=StrEnum)
_DistributionT = TypeVar("_DistributionT", bound="BaseDistribution")

LOGGER = logging.getLogger(__name__)


class BaseParametricStatistics(
    BaseStatistics,
    Generic[
        _DistributionT,
        _DefaultFittingCriterionT,
        _DistributionNameT,
        _FittingCriterionT,
        _SignificanceTestT,
    ],
):
    """Base class to compute statistics using probability distribution-fitting."""

    class _Distribution(NamedTuple):
        """A probability distribution."""

        name: str
        """The name of the probability distribution."""

        value: _DistributionT
        """The probability distribution."""

    _DistributionType = dict[str, Union[str, _DistributionT]]

    DistributionName: ClassVar[StrEnum] = _DistributionNameT
    FittingCriterion: ClassVar[StrEnum] = _FittingCriterionT
    SignificanceTest: ClassVar[StrEnum] = _SignificanceTestT
    SelectionCriterion: ClassVar[StrEnum] = BaseDistributionFitter.SelectionCriterion

    _DISTRIBUTION_FITTER: ClassVar[type[BaseDistributionFitter]]
    """The distribution fitter class."""

    fitting_criterion: _FittingCriterionT
    """The goodness-of-fit criterion, measuring how the distribution fits the data."""

    level: float
    """The test level used by the selection criteria that are significance tests.

    In statistical hypothesis testing, the test level corresponds to the risk of
    committing a type 1 error, that is an incorrect rejection of the null hypothesis
    """

    selection_criterion: SelectionCriterion
    """The selection criterion to select a distribution from a list of candidates."""

    distributions: dict[str, _DistributionType | list[_DistributionType]]
    """The probability distributions of the random variables.

    When a random variable is a random vector, its probability distribution is expressed
    as a list of marginal distributions. Otherwise, its probability distribution is
    expressed as the unique marginal distribution.
    """

    def __init__(
        self,
        dataset: Dataset,
        distributions: Sequence[_DistributionNameT],
        variable_names: Iterable[str] = (),
        fitting_criterion: _FittingCriterionT | None = None,
        level: float = 0.05,
        selection_criterion: SelectionCriterion = SelectionCriterion.BEST,
        name: str = "",
    ) -> None:
        """
        Args:
            distributions: The names of the probability distributions.
            fitting_criterion: The name of the fitting criterion
                to measure the goodness-of-fit of the probability distributions.
                If empty, use the default one.
                Use :meth:`.get_criteria` to get the available criteria.
            level: A test level,
                i.e. the risk of committing a Type 1 error,
                that is an incorrect rejection of a true null hypothesis,
                for criteria based on test hypothesis.
            selection_criterion: The name of the criterion
                to select a distribution among ``distributions``.
        """  # noqa: D205,D212,D415
        super().__init__(dataset, variable_names, name)
        self.fitting_criterion = (
            fitting_criterion or self._DISTRIBUTION_FITTER.default_fitting_criterion
        )
        self.selection_criterion = selection_criterion
        LOGGER.info("| Set goodness-of-fit criterion: %s.", fitting_criterion)
        self.level = level
        if self.fitting_criterion in self.SignificanceTest.__members__:
            LOGGER.info("| Set significance level of hypothesis test: %s.", level)

        self._all_distributions = self._fit_distributions(distributions)
        self.__distributions = {}
        self.distributions = self._select_best_distributions(distributions)

    def get_fitting_matrix(self) -> str:
        """Get the fitting matrix.

        This matrix contains goodness-of-fit measures
        for each pair < variable, distribution >.

        Returns:
            The printable fitting matrix.
        """
        variables = sorted(self._all_distributions.keys())
        distributions = list(self._all_distributions[variables[0]][0].keys())
        table = PrettyTable(["Variable", *distributions, "Selection"])
        for variable in variables:
            for index in range(self.dataset.variable_names_to_n_components[variable]):
                row = (
                    [variable]
                    + [
                        str(self.get_criteria(variable, index)[0][distribution])
                        for distribution in distributions
                    ]
                    + [self.__distributions[variable][index].name]
                )
                table.add_row(row)
        return str(table)

    def get_criteria(
        self, variable: str, index: int = 0
    ) -> tuple[dict[str, float], bool]:
        """Get the value of the fitting criterion for the different distributions.

        Args:
            variable: The name of the variable.
            index: The component of the variable.

        Returns:
            The value of the fitting criterion for the given variable name and component
            and the different distributions,
            as well as whether this fitting criterion is a statistical test
            and so this value a p-value.
        """
        distribution_names_to_criterion_values = {
            name: result["criterion"]
            for name, result in self._all_distributions[variable][index].items()
        }
        criterion_value_is_p_value = False
        if self.fitting_criterion in self.SignificanceTest.__members__:
            distribution_names_to_criterion_values = {
                name: result[1]["p-value"]
                for name, result in distribution_names_to_criterion_values.items()
            }
            criterion_value_is_p_value = True

        return distribution_names_to_criterion_values, criterion_value_is_p_value

    def plot_criteria(
        self,
        variable: str,
        title: str = "",
        save: bool = False,
        show: bool = True,
        directory: str | Path = ".",
        index: int = 0,
        fig_size: FigSizeType = (6.4, 3.2),
    ) -> Figure:
        """Plot criteria for a given variable name.

        Args:
            variable: The name of the variable.
            title: The title of the plot, if any.
            save: If ``True``, save the plot on the disk.
            show: If ``True``, show the plot.
            directory: The directory path, either absolute or relative.
            index: The index of the component of the variable.
            fig_size: The width and height of the figure in inches, e.g. ``(w, h)``.

        Raises:
            ValueError: If the variable is missing from the dataset.
        """
        if variable not in self.names:
            msg = (
                f"The variable '{variable}' is missing from the dataset; "
                f"available ones are: {pretty_str(self.names)}."
            )
            raise ValueError(msg)
        criteria, is_p_value = self.get_criteria(variable, index)
        x_values = []
        y_values = []
        labels = []
        x_value = 0
        for distribution, criterion in criteria.items():
            x_value += 1  # noqa: SIM113
            x_values.append(x_value)
            y_values.append(criterion)
            labels.append(distribution)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size)
        ax1.bar(x_values, y_values, tick_label=labels)
        if is_p_value:
            ax1.set_title(f"{self.fitting_criterion} (p-value)")
            ax1.axhline(self.level, color="r", linewidth=2.0)
        else:
            ax1.set_title(self.fitting_criterion)

        ax1.grid(True, "both")
        ax1.set_box_aspect(1)
        ax1.set_xlabel("Probability distributions")

        data = self.dataset.get_view(variable_names=variable).to_numpy()
        data_min = min(data)
        data_max = max(data)
        x_values = linspace(data_min, data_max, 1000)
        distributions = self._all_distributions[variable][index]
        ax2.hist(data, density=True)
        for dist_name, dist_value in distributions.items():
            pdf = dist_value["fitted_distribution"].distribution.computePDF
            y_values = [pdf([x_value])[0] for x_value in x_values]
            ax2.plot(x_values, y_values, label=dist_name, linewidth=2.0)

        ax2.set_box_aspect(1)
        ax2.legend()
        ax2.grid(True, "both")
        ax2.set_title("Probability density function")
        ax2.set_xlabel(
            repr_variable(
                variable, index, self.dataset.variable_names_to_n_components[variable]
            )
        )
        if title:
            plt.suptitle(title)

        save_show_figure(fig, show, Path(directory) / "criteria.pdf" if save else "")

        return fig

    def _select_best_distributions(
        self, distribution_names: Sequence[_DistributionNameT]
    ) -> dict[str, _DistributionType | list[_DistributionType]]:
        """Select the best distributions for the different variables.

        Args:
            distribution_names: The distribution names.

        Returns:
            The best distributions for the different variables.
        """
        LOGGER.info("Select the best distribution for each variable.")
        distributions = {}
        select_from_measures = self._DISTRIBUTION_FITTER.select_from_measures
        for variable in self.names:
            selected_distribution_names = []
            marginal_distributions = []
            for component, all_distributions in enumerate(
                self._all_distributions[variable]
            ):
                distribution_name = distribution_names[
                    select_from_measures(
                        [
                            all_distributions[distribution]["criterion"]
                            for distribution in distribution_names
                        ],
                        self.fitting_criterion,
                        self.level,
                        self.selection_criterion,
                    )
                ]
                best_dist = all_distributions[distribution_name]["fitted_distribution"]
                selected_distribution_names.append(distribution_name)
                marginal_distributions.append(best_dist)
                LOGGER.info(
                    "| The best distribution for %s[%s] is %s.",
                    variable,
                    component,
                    best_dist,
                )

            self.__distributions[variable] = list(
                map(
                    self._Distribution,
                    selected_distribution_names,
                    marginal_distributions,
                )
            )
            if len(marginal_distributions) == 1:
                distributions[variable] = self.__distributions[variable][0]
            else:
                distributions[variable] = self.__distributions[variable]

        return distributions

    def _fit_distributions(
        self,
        distributions: Iterable[_DistributionNameT],
    ) -> dict[str, list[dict[str, dict[str, _DistributionT | MeasureType]]]]:
        """Fit different distributions for the different marginals.

        Args:
            distributions: The distributions names.

        Returns:
            The distributions for the different variables.
        """
        LOGGER.info(
            "Fit different distributions (%s) per variable "
            "and compute the goodness-of-fit criterion.",
            ", ".join(distributions),
        )
        results = {}
        for name in self.names:
            LOGGER.info("| Fit different distributions for %s.", name)
            dataset_values = self.dataset.get_view(variable_names=name).to_numpy()
            size = self.dataset.variable_names_to_n_components[name]
            results[name] = [
                self._fit_marginal_distributions(
                    repr_variable(name, index, size), column, distributions
                )
                for index, column in enumerate(dataset_values.T)
            ]
        return results

    def _fit_marginal_distributions(
        self,
        variable: str,
        sample: RealArray,
        distributions: Iterable[_DistributionNameT],
    ) -> dict[str, dict[str, _DistributionT | MeasureType]]:
        """Fit different distributions for a given dataset marginal.

        Args:
            variable: A variable name.
            sample: A data array.
            distributions: The names of the distributions.

        Returns:
            The distributions for the different variables.
        """
        factory = self._DISTRIBUTION_FITTER(variable, sample)
        result = {}
        for distribution in distributions:
            fitted_distribution = factory.fit(distribution)
            result[distribution] = {
                "fitted_distribution": fitted_distribution,
                "criterion": factory.compute_measure(
                    fitted_distribution, self.fitting_criterion, self.level
                ),
            }
        return result

    def compute_maximum(self) -> dict[str, RealArray]:  # noqa: D102
        return {
            name: array([
                distribution.value.math_upper_bound
                for distribution in self.__distributions[name]
            ])
            for name in self.names
        }

    def compute_mean(self) -> dict[str, RealArray]:  # noqa: D102
        return {
            name: array([
                distribution.value.mean for distribution in self.__distributions[name]
            ])
            for name in self.names
        }

    def compute_minimum(self) -> dict[str, RealArray]:  # noqa: D102
        return {
            name: array([
                distribution.value.math_lower_bound
                for distribution in self.__distributions[name]
            ])
            for name in self.names
        }

    def compute_probability(  # noqa: D102
        self, thresh: Mapping[str, float | RealArray], greater: bool = True
    ) -> dict[str, RealArray]:
        func = lambda x: 1 - x if greater else x  # noqa: E731
        new_thresh = {}
        for name, value in thresh.items():
            if isinstance(value, float):
                new_thresh[name] = [
                    value
                ] * self.dataset.variable_names_to_n_components[name]
            elif len(value) == 1:
                new_thresh[name] = [
                    value[0]
                ] * self.dataset.variable_names_to_n_components[name]
            else:
                new_thresh[name] = value

        return {
            name: array([
                func(distribution.value.compute_cdf(new_thresh[name][index]))
                for index, distribution in enumerate(self.__distributions[name])
            ])
            for name in self.names
        }

    def compute_joint_probability(  # noqa: D102
        self, thresh: Mapping[str, float | RealArray], greater: bool = True
    ) -> dict[str, float]:
        raise NotImplementedError

    def compute_tolerance_interval(  # noqa: D102
        self,
        coverage: float,
        confidence: float = 0.95,
        side: BaseToleranceInterval.ToleranceIntervalSide = BaseToleranceInterval.ToleranceIntervalSide.BOTH,  # noqa:E501
    ) -> dict[str, list[BaseToleranceInterval.Bounds]]:
        if not 0.0 <= coverage <= 1.0:
            msg = "The argument 'coverage' must be a number in [0,1]."
            raise ValueError(msg)

        if not 0.0 <= confidence <= 1.0:
            msg = "The argument 'confidence' must be a number in [0,1]."
            raise ValueError(msg)

        tolerance_interval_factory = ToleranceIntervalFactory()
        return {
            name: [
                tolerance_interval_factory.get_class(distribution.name)(
                    self.n_samples,
                    *distribution.value.distribution.getParameter(),
                ).compute(coverage, confidence, side)
                for distribution in self.__distributions[name]
            ]
            for name in self.names
        }

    def compute_quantile(self, prob: float) -> dict[str, RealArray]:  # noqa: D102
        prob = array([prob])
        return {
            name: array([
                distribution.value.compute_inverse_cdf(prob)[0]
                for distribution in self.__distributions[name]
            ])
            for name in self.names
        }

    def compute_standard_deviation(self) -> dict[str, RealArray]:  # noqa: D102
        return {
            name: array([
                distribution.value.standard_deviation
                for distribution in self.__distributions[name]
            ])
            for name in self.names
        }

    def compute_variance(self) -> dict[str, RealArray]:  # noqa: D102
        return {
            name: array([
                distribution.value.standard_deviation**2
                for distribution in self.__distributions[name]
            ])
            for name in self.names
        }

    def compute_moment(self, order: int) -> dict[str, RealArray]:  # noqa: D102
        return {
            name: array([
                distribution.value.distribution.getMoment(order)[0]
                for distribution in self.__distributions[name]
            ])
            for name in self.names
        }

    def compute_range(self) -> dict[str, RealArray]:  # noqa: D102
        return {
            name: array([
                distribution.value.math_upper_bound
                - distribution.value.math_lower_bound
                for distribution in self.__distributions[name]
            ])
            for name in self.names
        }

    def plot(
        self,
        save: bool = False,
        show: bool = True,
        directory_path: str | Path = "",
        file_format: str = "png",
    ) -> dict[str, Figure]:
        """Visualize the cumulative distribution and probability density functions.

        Args:
            save: Whether to save the figures.
            show: Whether to show the figures.
            directory_path: The path to save the figures.
            file_format: The file extension.

        Returns:
            The cumulative distribution and probability density functions
            for each variable.
        """
        plots = {}
        for name in self.names:
            size = self.dataset.variable_names_to_n_components[name]
            for index, distribution in enumerate(self.__distributions[name]):
                plots[repr_variable(name, index, size)] = distribution.value.plot(
                    save=save,
                    show=show,
                    directory_path=directory_path,
                    file_extension=file_format,
                )

        return plots
