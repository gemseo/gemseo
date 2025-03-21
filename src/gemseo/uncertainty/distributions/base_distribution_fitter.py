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
"""Fitting a probability distribution to data using a UQ library."""

from __future__ import annotations

import logging
from abc import abstractmethod
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import MutableSequence
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Generic
from typing import TypeVar
from typing import Union

from strenum import StrEnum

from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta

if TYPE_CHECKING:
    from gemseo.typing import RealArray
    from gemseo.typing import StrKeyMapping

LOGGER = logging.getLogger(__name__)

FittingTestResultType = tuple[bool, Mapping[str, float]]
MeasureType = Union[FittingTestResultType, float]
_DistributionT = TypeVar("_DistributionT")


class BaseDistributionFitter(
    Generic[_DistributionT], metaclass=ABCGoogleDocstringInheritanceMeta
):
    """Base class to fit a probability distribution from data using a UQ library."""

    _data: RealArray
    """The data array."""

    _samples: Any
    """The samples."""

    _CRITERIA_TO_WRAPPED_OBJECTS: ClassVar[StrKeyMapping]
    """Fitting criteria to objects of the UQ library."""

    DistributionName: ClassVar[StrEnum]
    """The names of the probability distributions in the UQ library."""

    FittingCriterion: ClassVar[StrEnum]
    """The names of the fitting criteria."""

    SignificanceTest: ClassVar[StrEnum]
    """The names of the fitting criteria that are statistical significance tests."""

    class SelectionCriterion(StrEnum):
        """The selection criteria."""

        FIRST = "first"
        """Select the first distribution satisfying a fitting criterion."""

        BEST = "best"
        """Select the distribution that best satisfies a fitting criterion"""

    _FITTING_CRITERIA_TO_MINIMIZE: ClassVar[set[str]] = set()
    """The fitting criteria to minimize (the others are to be maximized)."""

    def __init__(self, data: RealArray) -> None:
        """
        Args:
            data: A data array.
        """  # noqa: D205,D212,D415
        self.data = data

    @property
    def data(self) -> RealArray:
        """The data array."""
        return self._data

    @data.setter
    def data(self, data_: RealArray) -> None:
        self._data = data_
        self._samples = data_.ravel()

    @abstractmethod
    def fit(
        self,
        distribution: DistributionName,  # noqa: F821
    ) -> _DistributionT:
        """Fit a probability distribution to the data.

        Args:
            distribution: The name of a probability distribution in the UQ library.

        Returns:
            The probability distribution fitted to the data.
        """

    def compute_measure(
        self,
        distribution: _DistributionT | DistributionName,  # noqa: F821
        criterion: FittingCriterion,  # noqa: F821
        level: float = 0.05,
    ) -> MeasureType:
        """Measure a goodness-of-fit of a probability distribution fitted to data.

        Args:
            distribution: Either a |g| probability distribution fitted to :attr:`.data`
                or the name of a probability distribution in the UQ library.
            criterion: The name of the fitting criterion
                to measure the goodness-of-fit of the probability distribution.
            level: A test level,
                i.e. the risk of committing a Type 1 error,
                that is an incorrect rejection of a true null hypothesis,
                for criteria based on test hypothesis.

        Returns:
            The goodness-of-fit of the probability distribution fitted to data.
        """
        goodness_of_fit = self._compute_measure(distribution, criterion, level)
        if criterion in {t.value for t in self.SignificanceTest}:
            return self._format_significance_test_goodness_of_fit(
                goodness_of_fit, level
            )

        return goodness_of_fit

    @abstractmethod
    def _compute_measure(
        self,
        distribution: _DistributionT | DistributionName,  # noqa: F821
        criterion: FittingCriterion,  # noqa: F821
        level: float,
    ) -> Any:
        """Compute a goodness-of-fit of a probability distribution fitted to data.

        This method does not format the result,
        unlike its caller :meth:`.compute_measure`.

        Args:
            distribution: Either a |g| probability distribution fitted to :attr:`.data`
                or the name of a probability distribution in the UQ library.
            criterion: The name of the fitting criterion
                to measure the goodness-of-fit of the probability distribution.
            level: A test level,
                i.e. the risk of committing a Type 1 error,
                that is an incorrect rejection of a true null hypothesis,
                for criteria based on test hypothesis.

        Returns:
            The unformatted goodness-of-fit
            of the probability distribution fitted to data.
        """

    @staticmethod
    def _format_significance_test_goodness_of_fit(
        goodness_of_fit: Any, level: float
    ) -> FittingTestResultType:
        """Format a goodness-of-fit measured according to a fitting criterion.

        Args:
            goodness_of_fit: The goodness-of-fit
                measured according to a fitting criterion.

        Returns:
            First,
            whether the null hypothesis is accepted,
            then,
            a dictionary whose keys are "p-value", "statistics" and "level".
        """

    def select(
        self,
        distributions: MutableSequence[_DistributionT | DistributionName],  # noqa: F821
        fitting_criterion: FittingCriterion,  # noqa: F821
        level: float = 0.05,
        selection_criterion: SelectionCriterion = SelectionCriterion.BEST,
    ) -> _DistributionT:
        """Select the best probability distribution according to a fitting criterion.

        Args:
            distributions: A collection of |g| probability distributions
                fitted to :attr:`.data`
                or names of probability distributions in the UQ library.
            fitting_criterion: The name of the fitting criterion
                to measure the goodness-of-fit of the probability distribution.
            level: A test level,
                i.e. the risk of committing a Type 1 error,
                that is an incorrect rejection of a true null hypothesis,
                for criteria based on test hypothesis.
            selection_criterion: The name of the selection criterion.

        Returns:
            The best probability distribution
            according to the fitting criterion and the selection criterion.
        """
        measures = []
        for index, distribution in enumerate(distributions):
            if distribution in self.DistributionName.__members__:
                distribution = self.fit(distribution)

            distributions[index] = distribution
            measures.append(
                self.compute_measure(distribution, fitting_criterion, level)
            )

        best_distribution_index = self.select_from_measures(
            measures, fitting_criterion, level, selection_criterion
        )
        return distributions[best_distribution_index]

    @classmethod
    def select_from_measures(
        cls,
        measures: MutableSequence[MeasureType],
        fitting_criterion: FittingCriterion,  # noqa: F821
        level: float = 0.05,
        selection_criterion: SelectionCriterion = SelectionCriterion.BEST,
    ) -> int:
        """Select the best probability distribution according to a fitting criterion.

        Args:
            measures: The goodness-of-fit measures.
            fitting_criterion: The name of the fitting criterion
                to measure the goodness-of-fit of the probability distribution.
            level: A test level,
                i.e. the risk of committing a Type 1 error,
                that is an incorrect rejection of a true null hypothesis,
                for criteria based on test hypothesis.
            selection_criterion: The name of the selection criterion.

        Returns:
            The index of the best probability distribution
            according to the fitting criterion and the selection criterion.
        """
        is_significant_test = fitting_criterion in cls.SignificanceTest.__members__
        if is_significant_test:
            measures = [measure[1]["p-value"] for index, measure in enumerate(measures)]
            if all(p_value < level for p_value in measures):
                LOGGER.warning(
                    "All criteria values are lower than the significance level %s.",
                    level,
                )

        if (
            selection_criterion == cls.SelectionCriterion.BEST
            or not is_significant_test
        ):
            return cls.__compute_index(fitting_criterion, measures)

        for index, measure in enumerate(measures):
            if measure >= level:
                return index

        return cls.__compute_index(fitting_criterion, measures)

    @classmethod
    def __compute_index(
        cls,
        fitting_criterion: FittingCriterion,  # noqa: F821
        measures: Iterable[MeasureType],
    ) -> int:
        """Compute the best distribution index according to a fitting criterion.

        Args:
            fitting_criterion: The name of the fitting criterion
                to measure the goodness-of-fit of the probability distribution.
            measures: The goodness-of-fit measures.

        Returns:
            The index of the best probability distribution
            according to a fitting criterion.
        """
        op = min if fitting_criterion in cls._FITTING_CRITERIA_TO_MINIMIZE else max
        return measures.index(op(measures))

    @property
    def available_distributions(self) -> list[str]:
        """The available probability distributions."""
        return sorted({t.value for t in self.DistributionName})

    @property
    def available_criteria(self) -> list[str]:
        """The available fitting criteria."""
        return sorted({t.value for t in self.FittingCriterion})

    @property
    def available_significance_tests(self) -> list[str]:
        """The significance tests."""
        return sorted({t.value for t in self.SignificanceTest})
