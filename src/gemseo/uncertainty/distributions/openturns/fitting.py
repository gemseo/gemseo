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
"""Class to fit a distribution from data based on OpenTURNS.

Overview
--------

The :class:`.OTDistributionFitter` class considers several samples
of an uncertain variable, fits a user-defined probability distribution
from this dataset and returns a :class:`.OTDistribution`.
It can also return a goodness-of-fit measure
associated with this distribution,
e.g. Bayesian Information Criterion, Kolmogorov test or Chi Squared test,
or select an optimal distribution among a collection according to
a criterion with a threshold.

Construction
------------

The :class:`.OTDistributionFitter` of a given uncertain variable is built
from only two arguments:

- a variable name,
- a one-dimensional numpy array.

Capabilities
------------

Fit a distribution
~~~~~~~~~~~~~~~~~~

The :meth:`.OTDistributionFitter.fit` method takes a distribution name
recognized by OpenTURNS as argument (e.g. 'Normal', 'Uniform', 'Exponential',
...) as argument and returns an :class:`.OTDistribution`
whose underlying OpenTURNS distribution is the specified one fitted
from the dataset passed to the constructor.

Measure the goodness-of-fit
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :meth:`.OTDistributionFitter.measure` method has two mandatory arguments:

- a distribution which is either a :class:`.OTDistribution`
  or a distribution name from which :meth:`!fit` method
  builds a :class:`.OTDistribution`,
- a fitting criterion name.

.. note::

   Use the :meth:`.OTDistributionFitter.get_available_criteria` method to get
   the complete list of available criteria
   and the :meth:`.OTDistributionFitter.get_significance_tests` method
   to get the list of available criteria which are significance tests.

The :meth:`.OTDistributionFitter.measure` method can also use a level
associated with the criterion.

The :meth:`.OTDistributionFitter.measure` methods returns a goodness-of-fit
measure whose nature is either a scalar
when the criterion is not a significance test
or a tuple when the criterion is a significance test. In that case,
the first component of the tuple is a boolean indicating if the measured
distribution is acceptable to model the data and the second one is
a dictionary containing the test statistics, the p-value and
the significance level.

Select an optimal distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :meth:`.OTDistributionFitter.select` method select aims to select an
optimal distribution among a collection. It uses two mandatory arguments:

- a list of distribution, either a list of distributions names
  or a list of :class:`.OTDistribution`,
- a fitting criterion name.

The :meth:`.OTDistributionFitter.select` method can also use a level
associated with the criterion and a criterion selection:

- 'best': select the distribution minimizing (or maximizing, depending
  on the criterion) the criterion,
- 'first': Select the first distribution for which the criterion is
  greater (or lower, depending on the criterion) than the level.
"""
from __future__ import annotations

import logging
from typing import Callable
from typing import Mapping
from typing import Sequence
from typing import Tuple
from typing import Union

import openturns as ots
from numpy import ndarray

from gemseo.uncertainty.distributions.openturns.distribution import OTDistribution

LOGGER = logging.getLogger(__name__)

MeasureType = Union[Tuple[bool, Mapping[str, float]], float]


class OTDistributionFitter:
    """Fit a probabilistic distribution from a data array."""

    variable: str
    """The name of the variable."""

    data: ndarray
    """The data array."""

    _AVAILABLE_DISTRIBUTIONS = {}
    for factory in ots.DistributionFactory.GetContinuousUniVariateFactories():
        factory_class_name = factory.getImplementation().getClassName()
        dist_name = factory_class_name.split("Factory")[0]
        _AVAILABLE_DISTRIBUTIONS[dist_name] = getattr(ots, factory_class_name)

    AVAILABLE_DISTRIBUTIONS = sorted(_AVAILABLE_DISTRIBUTIONS.keys())

    _AVAILABLE_FITTING_TESTS = {
        "BIC": ots.FittingTest.BIC,
        "Kolmogorov": ots.FittingTest.Kolmogorov,
        "ChiSquared": ots.FittingTest.ChiSquared,
    }
    AVAILABLE_FITTING_TESTS = sorted(_AVAILABLE_FITTING_TESTS.keys())

    SIGNIFICANCE_TESTS = ["Kolmogorov", "ChiSquared"]

    _CRITERIA_TO_MINIMIZE = ["BIC"]

    def __init__(
        self,
        variable: str,
        data: ndarray,
    ) -> None:
        """
        Args:
            variable: The name of the variable.
            data: A data array.
        """  # noqa: D205,D212,D415
        self.variable = variable
        try:
            isinstance(data, ndarray)
            self.data = ots.Sample(data.reshape((-1, 1)))
        except AttributeError:
            raise TypeError("data must be a numpy array")

    def _get_factory(
        self,
        distribution: str,
    ) -> ots.DistributionFactory:
        """Get the distribution factory.

        Args:
            distribution: The name of the distribution.

        Returns:
            An OpenTURNS distribution factory.
        """
        try:
            distribution_factory = self._AVAILABLE_DISTRIBUTIONS[distribution]
        except KeyError:
            distributions = ", ".join(list(self._AVAILABLE_DISTRIBUTIONS.keys()))
            raise ValueError(
                "{} is not a name of distribution available for fitting; "
                "available ones are: {}.".format(distribution, distributions)
            )
        return distribution_factory

    def _get_fitting_test(
        self,
        criterion: str,
    ) -> Callable:
        """Get the fitting test.

        Args:
            criterion: The name of a fitting criterion.

        Returns:
            The OpenTURNS fitting test corresponding to the provided name.
        """
        try:
            fitting_test = self._AVAILABLE_FITTING_TESTS[criterion]
        except KeyError:
            tests = ", ".join(list(self._AVAILABLE_FITTING_TESTS.keys()))
            raise ValueError(
                "{} is not a name of fitting test; "
                "available ones are: {}.".format(criterion, tests)
            )
        return fitting_test

    def fit(
        self,
        distribution: str,
    ) -> OTDistribution:
        """Fit a distribution.

        Args:
            distribution: The name of a distribution.

        Returns:
            The distribution corresponding to the provided name.
        """
        factory = self._get_factory(distribution)
        fitted_distribution = factory().build(self.data)
        parameters = fitted_distribution.getParameter()
        distribution = OTDistribution(self.variable, distribution, parameters)
        return distribution

    def compute_measure(
        self,
        distribution: OTDistribution | str,
        criterion: str,
        level: float = 0.05,
    ) -> MeasureType:
        """Measure the goodness-of-fit of a distribution to data.

        Args:
            distribution: A distribution name.
            criterion: The name of the goodness-of-fit criterion.
            level: A test level,
                i.e. the risk of committing a Type 1 error,
                that is an incorrect rejection of a true null hypothesis,
                for criteria based on test hypothesis.

        Returns:
            The goodness-of-fit measure.
        """
        if isinstance(distribution, str):
            distribution = self.fit(distribution)
        if distribution.dimension > 1:
            raise TypeError("A 1D distribution is required.")
        distribution = distribution.marginals[0]
        fitting_test = self._get_fitting_test(criterion)
        if criterion in self.SIGNIFICANCE_TESTS:
            result = fitting_test(self.data, distribution, level)
            details = {
                "p-value": result.getPValue(),
                "statistics": result.getStatistic(),
                "level": level,
            }
            result = (result.getBinaryQualityMeasure(), details)
        else:
            result = fitting_test(self.data, distribution)
        return result

    def select(
        self,
        distributions: Sequence[str] | Sequence[OTDistribution],
        fitting_criterion: str,
        level: float = 0.05,
        selection_criterion: str = "best",
    ) -> OTDistribution:
        """Select the best distribution from a list of candidates.

        Args:
            distributions: The distributions.
            fitting_criterion: The name of the goodness-of-fit criterion.
            level: A test level,
                i.e. the risk of committing a Type 1 error,
                that is an incorrect rejection of a true null hypothesis,
                for criteria based on test hypothesis.
            selection_criterion:  The name of the selection criterion.
                Either 'first' or 'best'.

        Returns:
            The best distribution.
        """
        measures = []
        for index, distribution in enumerate(distributions):
            if isinstance(distribution, str):
                distribution = self.fit(distribution)
            measures.append(
                self.compute_measure(distribution, fitting_criterion, level)
            )
            distributions[index] = distribution
        index = self.select_from_measures(
            measures, fitting_criterion, level, selection_criterion
        )
        return distributions[index]

    @classmethod
    def select_from_measures(
        cls,
        measures: list[MeasureType],
        fitting_criterion: str,
        level: float = 0.05,
        selection_criterion: str = "best",
    ) -> int:
        """Select the best distribution from measures.

        Args:
            measures: The measures.
            fitting_criterion: The name of the goodness-of-fit criterion.
            level: A test level,
                i.e. the risk of committing a Type 1 error,
                that is an incorrect rejection of a true null hypothesis,
                for criteria based on test hypothesis.
            selection_criterion:  The name of the selection criterion.
                Either 'first' or 'best'.

        Returns:
            The index of the best distribution.
        """
        if fitting_criterion in cls.SIGNIFICANCE_TESTS:
            for index, _ in enumerate(measures):
                measures[index] = measures[index][1]["p-value"]
            if sum(p_value > level for p_value in measures) == 0:
                LOGGER.warning(
                    "All criteria values are lower than the significance level %s.",
                    level,
                )
        if selection_criterion == "best" or level is None:
            index = cls.__find_opt_distribution(measures, fitting_criterion)
        else:
            index = cls.__apply_first_strategy(measures, fitting_criterion, level)
        return index

    @classmethod
    def __apply_first_strategy(
        cls,
        measures: list[float],
        fitting_criterion: str,
        level: float = 0.05,
    ) -> int:
        """Select the best distribution from measures by applying the "first" strategy.

        Args:
            measures: The measures.
            fitting_criterion: The name of the goodness-of-fit criterion.
            level: A test level,
                i.e. the risk of committing a Type 1 error,
                that is an incorrect rejection of a true null hypothesis,
                for criteria based on test hypothesis.

        Returns:
            The index of the best distribution.
        """
        select = False
        index = 0
        for measure in measures:
            select = measure >= level
            if select:
                break
            index += 1
        if not select:
            index = cls.__find_opt_distribution(measures, fitting_criterion)
        return index

    @classmethod
    def __find_opt_distribution(
        cls,
        measures: list[float],
        fitting_criterion: str,
    ) -> int:
        """Select the best distribution from measures by applying the "best" strategy.

        Args:
            measures: The measures.
            fitting_criterion: The name of the goodness-of-fit criterion.

        Returns:
            The index of the optimum distribution.
        """
        if fitting_criterion in cls._CRITERIA_TO_MINIMIZE:
            index = measures.index(min(measures))
        else:
            index = measures.index(max(measures))
        return index

    @property
    def available_distributions(self) -> list[str]:
        """The available distributions."""
        return sorted(self._AVAILABLE_DISTRIBUTIONS.keys())

    @property
    def available_criteria(self) -> list[str]:
        """The available goodness-of-fit criteria."""
        return sorted(self._AVAILABLE_FITTING_TESTS.keys())

    @property
    def available_significance_tests(self) -> list[str]:
        """The significance tests."""
        return sorted(self.SIGNIFICANCE_TESTS)
