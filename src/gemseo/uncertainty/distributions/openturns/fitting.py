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
from this dataset and returns an :class:`.OTDistribution`.
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

- a distribution which is either an :class:`.OTDistribution`
  or a distribution name from which :meth:`!fit` method
  builds an :class:`.OTDistribution`,
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
from collections.abc import Mapping
from collections.abc import MutableSequence
from collections.abc import Sequence
from typing import TYPE_CHECKING
from typing import Callable
from typing import ClassVar
from typing import Union

import openturns as ots
from strenum import LowercaseStrEnum
from strenum import StrEnum

from gemseo.uncertainty.distributions.openturns.distribution import OTDistribution

if TYPE_CHECKING:
    from numpy import ndarray

LOGGER = logging.getLogger(__name__)

MeasureType = Union[tuple[bool, Mapping[str, float]], float]


def _get_distribution_factories() -> dict[str, ots.DistributionFactory]:
    """Return the distribution factories.

    Returns:
        The mapping from the distributions to their factories.
    """
    dist_to_factory_class = {}
    for factory in ots.DistributionFactory.GetContinuousUniVariateFactories():
        factory_class_name = factory.getImplementation().getClassName()
        dist_name = factory_class_name.split("Factory")[0]
        dist_to_factory_class[dist_name] = getattr(ots, factory_class_name)
    return dist_to_factory_class


class OTDistributionFitter:
    """Fit a probabilistic distribution from a data array."""

    variable: str
    """The name of the variable."""

    data: ndarray
    """The data array."""

    _DISTRIBUTIONS_NAME_TO_FACTORY = _get_distribution_factories()

    _FITTINGS_CRITERION_TO_TEST: ClassVar[dict[str, ots.FittingTest]] = {
        "BIC": ots.FittingTest.BIC,
        "ChiSquared": ots.FittingTest.ChiSquared,
        "Kolmogorov": ots.FittingTest.Kolmogorov,
    }

    DistributionName = StrEnum(
        "DistributionName", sorted(_DISTRIBUTIONS_NAME_TO_FACTORY.keys())
    )
    """The available probability distributions."""

    FittingCriterion = StrEnum(
        "FittingCriterion", sorted(_FITTINGS_CRITERION_TO_TEST.keys())
    )
    """The available fitting criteria."""

    SignificanceTest = StrEnum("SignificanceTest", "ChiSquared Kolmogorov")
    """The available significance tests."""

    SelectionCriterion = LowercaseStrEnum("SelectionCriterion", "FIRST BEST")
    """The different selection criteria."""

    __CRITERIA_TO_MINIMIZE: ClassVar[list[str]] = [FittingCriterion.BIC]

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
        self.data = ots.Sample(data.reshape((-1, 1)))

    def _get_factory(
        self,
        distribution_name: DistributionName,
    ) -> ots.DistributionFactory:
        """Return the distribution factory.

        Args:
            distribution_name: The distribution name.

        Returns:
            The OpenTURNS distribution factory.
        """
        return self._DISTRIBUTIONS_NAME_TO_FACTORY[distribution_name]

    def _get_fitting_test(
        self,
        criterion: FittingCriterion,
    ) -> Callable:
        """Get the fitting test.

        Args:
            criterion: The fitting criterion.

        Returns:
            The OpenTURNS fitting test corresponding to the provided name.
        """
        return self._FITTINGS_CRITERION_TO_TEST[criterion]

    def fit(
        self,
        distribution: DistributionName,
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
        return OTDistribution(self.variable, distribution, parameters)

    def compute_measure(
        self,
        distribution: OTDistribution | DistributionName,
        criterion: FittingCriterion,
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
        if distribution in self.DistributionName.__members__:
            distribution = self.fit(distribution)
        if distribution.dimension > 1:
            raise TypeError("A 1D distribution is required.")
        distribution = distribution.marginals[0]
        fitting_test = self._get_fitting_test(criterion)
        if criterion in self.SignificanceTest.__members__:
            result = fitting_test(self.data, distribution, level)
            details = {
                "p-value": result.getPValue(),
                "statistics": result.getStatistic(),
                "level": level,
            }
            return result.getBinaryQualityMeasure(), details
        return fitting_test(self.data, distribution)

    def select(
        self,
        distributions: MutableSequence[DistributionName | OTDistribution],
        fitting_criterion: FittingCriterion,
        level: float = 0.05,
        selection_criterion: SelectionCriterion = SelectionCriterion.BEST,
    ) -> OTDistribution:
        """Select the best distribution from a list of candidates.

        Args:
            distributions: The distributions.
            fitting_criterion: The goodness-of-fit criterion.
            level: A test level,
                i.e. the risk of committing a Type 1 error,
                that is an incorrect rejection of a true null hypothesis,
                for criteria based on test hypothesis.
            selection_criterion: The selection criterion.

        Returns:
            The best distribution.
        """
        measures = []
        for index, distribution in enumerate(distributions):
            if distribution in self.DistributionName.__members__:
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
        measures: MutableSequence[MeasureType],
        fitting_criterion: FittingCriterion,
        level: float = 0.05,
        selection_criterion: SelectionCriterion = SelectionCriterion.BEST,
    ) -> int:
        """Select the best distribution from measures.

        Args:
            measures: The measures.
            fitting_criterion: The goodness-of-fit criterion.
            level: A test level,
                i.e. the risk of committing a Type 1 error,
                that is an incorrect rejection of a true null hypothesis,
                for criteria based on test hypothesis.
            selection_criterion: The selection criterion.

        Returns:
            The index of the best distribution.
        """
        if fitting_criterion in cls.SignificanceTest.__members__:
            for index, _ in enumerate(measures):
                measures[index] = measures[index][1]["p-value"]
            if sum(p_value > level for p_value in measures) == 0:
                LOGGER.warning(
                    "All criteria values are lower than the significance level %s.",
                    level,
                )
        if selection_criterion == cls.SelectionCriterion.BEST or level is None:
            return cls.__find_opt_distribution(measures, fitting_criterion)
        return cls.__apply_first_strategy(measures, fitting_criterion, level)

    @classmethod
    def __apply_first_strategy(
        cls,
        measures: Sequence[float],
        fitting_criterion: FittingCriterion,
        level: float = 0.05,
    ) -> int:
        """Select the best distribution from measures by applying the "first" strategy.

        Args:
            measures: The measures.
            fitting_criterion: The goodness-of-fit criterion.
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
        measures: Sequence[float],
        fitting_criterion: FittingCriterion,
    ) -> int:
        """Select the best distribution from measures.

        By applying the :attr:`.SelectionCriterion.BEST` strategy.

        Args:
            measures: The measures.
            fitting_criterion: The goodness-of-fit criterion.

        Returns:
            The index of the optimum distribution.
        """
        if fitting_criterion in cls.__CRITERIA_TO_MINIMIZE:
            return measures.index(min(measures))
        return measures.index(max(measures))

    @property
    def available_distributions(self) -> list[str]:
        """The available distributions."""
        return sorted(self._DISTRIBUTIONS_NAME_TO_FACTORY.keys())

    @property
    def available_criteria(self) -> list[str]:
        """The available goodness-of-fit criteria."""
        return sorted(self._FITTINGS_CRITERION_TO_TEST.keys())

    @property
    def available_significance_tests(self) -> list[str]:
        """The significance tests."""
        return sorted(self.SignificanceTest)
