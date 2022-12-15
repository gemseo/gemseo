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
"""Class for the parametric estimation of statistics from a dataset.

Overview
--------

The :class:`.ParametricStatistics` class inherits
from the abstract :class:`.Statistics` class
and aims to estimate statistics from a :class:`.Dataset`,
based on candidate parametric distributions calibrated from this :class:`.Dataset`.

For each variable,

1. the parameters of these distributions are calibrated from the :class:`.Dataset`,
2. the fitted parametric :class:`.Distribution` which is optimal
   in the sense of a goodness-of-fit criterion and a selection criterion
   is selected to estimate the statistics related to this variable.

The :class:`.ParametricStatistics` relies on the OpenTURNS library
through the :class:`.OTDistribution` and :class:`.OTDistributionFitter` classes.

Construction
------------

The :class:`.ParametricStatistics` is built from two mandatory arguments:

- a dataset,
- a list of distributions names,

and can consider optional arguments:

- a subset of variables names
  (by default, statistics are computed for all variables),
- a fitting criterion name
  (by default, BIC is used;
  see :attr:`.AVAILABLE_CRITERIA`
  and :attr:`.AVAILABLE_SIGNIFICANCE_TESTS`
  for more information),
- a level associated with the fitting criterion,
- a selection criterion:

  - 'best':
    select the distribution minimizing
    (or maximizing, depending on the criterion)
    the criterion,
  - 'first':
    select the first distribution
    for which the criterion is greater
    (or lower, depending on the criterion)
    than the level,

- a name for the :class:`.ParametricStatistics` object
  (by default, the name is the concatenation of 'ParametricStatistics'
  and the name of the :class:`.Dataset`).

Capabilities
------------

By inheritance,
a :class:`.ParametricStatistics` object has
the same capabilities as :class:`.Statistics`.
Additional ones are:

- :meth:`.get_fitting_matrix`:
  this method displays the values of the fitting criterion
  for the different variables
  and candidate probability distributions
  as well as the select probability distribution,
- :meth:`.plot_criteria`:
  this method plots the criterion values for a given variable.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable
from typing import Sequence

import matplotlib.pyplot as plt
from numpy import array
from numpy import linspace
from numpy import ndarray

from gemseo.core.dataset import Dataset
from gemseo.third_party.prettytable.prettytable import PrettyTable
from gemseo.uncertainty.distributions.openturns.distribution import OTDistribution
from gemseo.uncertainty.distributions.openturns.fitting import MeasureType
from gemseo.uncertainty.distributions.openturns.fitting import OTDistributionFitter
from gemseo.uncertainty.statistics.statistics import Statistics
from gemseo.uncertainty.statistics.tolerance_interval.distribution import (
    ToleranceIntervalFactory,
)
from gemseo.uncertainty.statistics.tolerance_interval.distribution import (
    ToleranceIntervalSide,
)
from gemseo.utils.matplotlib_figure import save_show_figure

LOGGER = logging.getLogger(__name__)


class ParametricStatistics(Statistics):
    """Parametric estimation of statistics.

    Examples:
        >>> from gemseo.api import (
        ...     create_discipline,
        ...     create_parameter_space,
        ...     create_scenario
        ... )
        >>> from gemseo.uncertainty.statistics.parametric import ParametricStatistics
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
        ...     "x2", "OTNormalDistribution", mu=0.5, sigma=2
        ... )
        >>>
        >>> scenario = create_scenario(
        ...     [discipline],
        ...     "DisciplinaryOpt",
        ...     "y1", parameter_space, scenario_type="DOE"
        ... )
        >>> scenario.execute({'algo': 'OT_MONTE_CARLO', 'n_samples': 100})
        >>>
        >>> dataset = scenario.export_to_dataset(opt_naming=False)
        >>>
        >>> statistics = ParametricStatistics(
        ...     dataset, ['Normal', 'Uniform', 'Triangular']
        ... )
        >>> fitting_matrix = statistics.get_fitting_matrix()
        >>> mean = statistics.mean()
    """

    fitting_criterion: str
    """The name of the goodness-of-fit criterion,
    measuring how the distribution fits the data."""

    level: float
    """The test level,
    i.e. risk of committing a Type 1 error,
    that is an incorrect rejection of a true null hypothesis,
    for criteria based on test hypothesis."""

    selection_criterion: str
    """The name of the selection criterion
    to select a distribution from a list of candidates."""

    distributions: dict[str, dict[str, OTDistribution]]
    """The probability distributions of the random variables."""

    AVAILABLE_DISTRIBUTIONS = sorted(
        OTDistributionFitter._AVAILABLE_DISTRIBUTIONS.keys()
    )

    AVAILABLE_CRITERIA = sorted(OTDistributionFitter._AVAILABLE_FITTING_TESTS.keys())
    AVAILABLE_SIGNIFICANCE_TESTS = sorted(OTDistributionFitter.SIGNIFICANCE_TESTS)

    def __init__(
        self,
        dataset: Dataset,
        distributions: Sequence[str],
        variables_names: Iterable[str] | None = None,
        fitting_criterion: str = "BIC",
        level: float = 0.05,
        selection_criterion: str = "best",
        name: str | None = None,
    ) -> None:
        """
        Args:
            distributions: The names of the distributions.
            fitting_criterion: The name of
                the goodness-of-fit criterion,
                measuring how the distribution fits the data.
                Use :meth:`.ParametricStatistics.get_criteria`
                to get the available criteria.
            level: A test level,
                i.e. the risk of committing a Type 1 error,
                that is an incorrect rejection of a true null hypothesis,
                for criteria based on test hypothesis.
            selection_criterion: The name of the selection criterion
                to select a distribution from a list of candidates.
                Either 'first' or 'best'.
        """  # noqa: D205,D212,D415
        super().__init__(dataset, variables_names, name)
        significance_tests = OTDistributionFitter.SIGNIFICANCE_TESTS
        self.fitting_criterion = fitting_criterion
        self.selection_criterion = selection_criterion
        LOGGER.info("| Set goodness-of-fit criterion: %s.", fitting_criterion)
        if self.fitting_criterion in significance_tests:
            self.level = level
            LOGGER.info("| Set significance level of hypothesis test: %s.", level)
        else:
            self.level = None
        self._all_distributions = None
        self.distributions = None
        self._build_distributions(distributions)

    def _build_distributions(
        self,
        distributions: Sequence[str],
    ) -> None:
        """Build distributions from distributions names.

        Args:
            distributions: The names of the distributions.
        """
        self._all_distributions = self._fit_distributions(distributions)
        self.distributions = self._select_best_distributions(distributions)

    def get_fitting_matrix(self) -> str:
        """Get the fitting matrix.

        This matrix contains goodness-of-fit measures
        for each pair < variable, distribution >.

        Returns:
            The printable fitting matrix.
        """
        variables = sorted(self._all_distributions.keys())
        distributions = list(self._all_distributions[variables[0]].keys())
        table = PrettyTable(["Variable"] + distributions + ["Selection"])
        for variable in variables:
            row, _ = self.get_criteria(variable)
            row = [variable] + [row[distribution] for distribution in distributions]
            row += [self.distributions[variable]["name"]]
            table.add_row(row)
        return str(table)

    def get_criteria(
        self,
        variable: str,
    ) -> tuple[dict[str, float], bool]:
        """Get criteria for a given variable name and the different distributions.

        Args:
            variable: The name of the variable.

        Returns:
            The criterion for the different distributions.
            and an indicator equal to True is the criterion is a p-value.
        """
        all_distributions = self._all_distributions[variable]
        criteria = {
            distribution: result["criterion"]
            for distribution, result in all_distributions.items()
        }
        is_p_value = False
        significance_tests = OTDistributionFitter.SIGNIFICANCE_TESTS
        if self.fitting_criterion in significance_tests:
            criteria = {
                distribution: result[1]["p-value"]
                for distribution, result in criteria.items()
            }
            is_p_value = True
        return criteria, is_p_value

    def plot_criteria(
        self,
        variable: str,
        title: str | None = None,
        save: bool = False,
        show: bool = True,
        n_legend_cols: int = 4,
        directory: str | Path = ".",
    ) -> None:
        """Plot criteria for a given variable name.

        Args:
            variable: The name of the variable.
            title: A plot title.
            save: If True, save the plot on the disk.
            show: If True, show the plot.
            n_legend_cols: The number of text columns in the upper legend.
            directory: The directory path, either absolute or relative.

        Raises:
            ValueError: If the variable is missing from the dataset.
        """
        if variable not in self.names:
            raise ValueError(
                "The variable '{}' is missing from the dataset."
                "Available ones are: {}.".format(variable, ", ".join(self.names))
            )
        criteria, is_p_value = self.get_criteria(variable)
        x_values = []
        y_values = []
        labels = []
        x_value = 0
        for distribution, criterion in criteria.items():
            x_value += 1
            x_values.append(x_value)
            y_values.append(criterion)
            labels.append(distribution)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 3.2))
        ax1.bar(x_values, y_values, tick_label=labels)
        if is_p_value:
            plt.ylabel(f"p-value from {self.fitting_criterion} test")
            plt.axhline(self.level, color="r", linewidth=2.0)
        ax1.grid(True, "both")
        ax1.set_box_aspect(1)
        data = array(self.dataset[variable])
        data_min = min(data)
        data_max = max(data)
        x_values = linspace(data_min, data_max, 1000)
        distributions = self._all_distributions[variable]
        ax2.hist(data, density=True)

        for dist_name, dist_value in distributions.items():
            pdf = dist_value["fitted_distribution"].distribution.computePDF
            y_values = [pdf([x_value])[0] for x_value in x_values]
            ax2.plot(x_values, y_values, label=dist_name, linewidth=2.0)

        ax2.set_box_aspect(1)
        ax2.legend()
        ax2.grid(True, "both")
        if title is not None:
            plt.suptitle(title)

        if save:
            file_path = Path(directory) / "criteria.pdf"
        else:
            file_path = None

        save_show_figure(fig, show, file_path)

    def _select_best_distributions(
        self, distributions_names: Sequence[str]
    ) -> dict[str, dict[str, str | OTDistribution]]:
        """Select the best distributions for the different variables.

        Args:
            distributions_names: The names of the distributions.

        Returns:
            The best distributions for the different variables.
        """
        LOGGER.info("Select the best distribution for each variable.")
        distributions = {}
        for variable in self.names:
            all_distributions = self._all_distributions[variable]
            criteria = [
                all_distributions[distribution]["criterion"]
                for distribution in distributions_names
            ]
            select_from_measures = OTDistributionFitter.select_from_measures
            index = select_from_measures(
                criteria, self.fitting_criterion, self.level, self.selection_criterion
            )
            name = distributions_names[index]
            value = all_distributions[name]["fitted_distribution"]
            distributions[variable] = {"name": name, "value": value}
            LOGGER.info("| The best distribution for %s is %s.", variable, value)
        return distributions

    def _fit_distributions(
        self,
        distributions: Iterable[str],
    ) -> dict[str, dict[str, dict[str, OTDistribution | MeasureType]]]:
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
        for variable in self.names:
            LOGGER.info("| Fit different distributions for %s.", variable)
            dataset = self.dataset[variable]
            results[variable] = self._fit_marginal_distributions(
                variable, dataset, distributions
            )
        return results

    def _fit_marginal_distributions(
        self,
        variable: str,
        sample: ndarray,
        distributions: Iterable[str],
    ) -> dict[str, dict[str, OTDistribution | MeasureType]]:
        """Fit different distributions for a given dataset marginal.

        Args:
            variable: A variable name.
            sample: A data array.
            distributions: The names of the distributions.

        Returns:
            The distributions for the different variables.
        """
        result = {}
        factory = OTDistributionFitter(variable, sample)
        for distribution in distributions:
            fitted_distribution = factory.fit(distribution)
            test_result = factory.compute_measure(
                fitted_distribution, self.fitting_criterion, self.level
            )
            result[distribution] = {}
            result[distribution]["fitted_distribution"] = fitted_distribution
            result[distribution]["criterion"] = test_result
        return result

    def compute_maximum(self) -> dict[str, ndarray]:  # noqa: D102
        result = {
            name: self.distributions[name]["value"].math_upper_bound
            for name in self.names
        }
        return result

    def compute_mean(self) -> dict[str, ndarray]:  # noqa: D102
        result = {name: self.distributions[name]["value"].mean for name in self.names}
        return result

    def compute_minimum(self) -> dict[str, ndarray]:  # noqa: D102
        result = {
            name: self.distributions[name]["value"].math_lower_bound
            for name in self.names
        }
        return result

    def compute_probability(  # noqa: D102
        self,
        thresh: float,
        greater: bool = True,
    ) -> dict[str, ndarray]:
        dist = self.distributions
        if greater:
            result = {
                name: 1 - dist[name]["value"].compute_cdf(thresh[name])[0]
                for name in self.names
            }
        else:
            result = {
                name: dist[name]["value"].compute_cdf(thresh[name])[0]
                for name in self.names
            }
        return result

    def compute_tolerance_interval(  # noqa: D102
        self,
        coverage: float,
        confidence: float = 0.95,
        side: ToleranceIntervalSide = ToleranceIntervalSide.BOTH,
    ) -> dict[str, tuple[ndarray, ndarray]]:

        if not 0.0 <= coverage <= 1.0:
            raise ValueError("The argument 'coverage' must be number in [0,1].")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("The argument 'confidence' must be number in [0,1].")
        limits = {}
        factory = ToleranceIntervalFactory()
        for variable in self.names:
            distribution = self.distributions[variable]
            cls = factory.get_class(distribution["name"])
            parameters = distribution["value"].marginals[0].getParameter()
            tolerance_interval = cls(self.n_samples, *parameters)
            limits[variable] = tolerance_interval.compute(coverage, confidence, side)
        return limits

    def compute_quantile(  # noqa: D102
        self,
        prob: float,
    ) -> dict[str, ndarray]:
        prob = array([prob])
        result = {
            name: self.distributions[name]["value"].compute_inverse_cdf(prob)
            for name in self.names
        }
        return result

    def compute_standard_deviation(  # noqa: D102
        self,
    ) -> dict[str, ndarray]:
        result = {
            name: self.distributions[name]["value"].standard_deviation
            for name in self.names
        }
        return result

    def compute_variance(self) -> dict[str, ndarray]:  # noqa: D102
        result = {
            name: self.distributions[name]["value"].standard_deviation ** 2
            for name in self.names
        }
        return result

    def compute_moment(  # noqa: D102
        self,
        order: int,
    ) -> dict[str, ndarray]:
        dist = self.distributions
        result = [
            dist[name]["value"].distribution.getMoment(order)[0] for name in self.names
        ]
        return result

    def compute_range(self) -> dict[str, ndarray]:  # noqa: D102
        result = {}
        for name in self.names:
            dist = self.distributions[name]["value"]
            result[name] = dist.math_upper_bound - dist.math_lower_bound
        return result
