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
"""The base class for the maximum-entropy distributions."""

from __future__ import annotations

import logging
from abc import abstractmethod
from math import inf
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Final

from gemseo.uncertainty.distribution._maximum_entropy.family import MaximumEntropyFamily
from gemseo.uncertainty.distribution._maximum_entropy.parameter import (
    MaximumEntropyParameter,
)
from gemseo.uncertainty.distribution._maximum_entropy.selection import (
    compute_maximum_entropy_moments,
)
from gemseo.uncertainty.distribution._maximum_entropy.selection import (
    compute_maximum_entropy_parameters,
)
from gemseo.util.metaclass import ABCGoogleDocstringInheritanceMeta
from gemseo.util.pydantic import create_model

if TYPE_CHECKING:
    from gemseo.uncertainty.distribution.core.base_settings import (
        BaseGenericDistributionSettings,
    )
    from gemseo.uncertainty.distribution.core.base_univariate_settings.maximum_entropy import (  # noqa: E501
        BaseMaximumEntropyDistributionSettings,
    )
    from gemseo.util.typing import StrKeyMapping

logger = logging.getLogger(__name__)

_distribution: Final[str] = "distribution"
"""The name of the standard parameter storing the maximum-entropy family."""

_information_names: Final[tuple[str, ...]] = ("minimum", "maximum", "mu", "sigma")
"""The names of the settings expressing what is known about the random variable."""


class BaseMaximumEntropyDistribution(metaclass=ABCGoogleDocstringInheritanceMeta):
    """The base class for the maximum-entropy distributions.

    The parameters of the probability distribution maximizing the entropy
    for the information expressed by the settings are computed
    independently of the library implementing the distribution.
    This class provides what the derived classes share
    to translate these parameters into the settings of a library-specific distribution.
    """

    settings_class: ClassVar[type[BaseMaximumEntropyDistributionSettings]]
    """The class of the settings of the maximum-entropy distribution."""

    __mean: float
    """The expectation of the random variable."""

    __standard_deviation: float
    """The standard deviation of the random variable."""

    def __init__(  # noqa: D107
        self, settings: BaseMaximumEntropyDistributionSettings | None = None
    ) -> None:
        settings = create_model(self.settings_class, settings_model=settings)
        family, parameters = compute_maximum_entropy_parameters(
            settings.minimum,
            settings.maximum,
            settings.mu,
            settings.sigma,
            settings=settings.numerical_settings,
        )
        family, parameters = self._substitute_extreme_truncated_normal(
            family, parameters, settings
        )
        if family == MaximumEntropyFamily.TRUNCATED_NORMAL:
            # The solver computing the parameters matches these moments.
            self.__mean = settings.mu
            self.__standard_deviation = settings.sigma
        else:
            self.__mean, self.__standard_deviation = compute_maximum_entropy_moments(
                family, parameters, settings=settings.numerical_settings
            )
        super().__init__(
            self._create_maximum_entropy_settings(family, parameters, settings)
        )
        self._set_support(settings)

    @property
    def mean(self) -> float:
        """The expectation of the random variable.

        It is computed from the closed-form expressions
        of the maximum-entropy distributions
        instead of being requested from the library,
        whose moment routines lose their accuracy for some parameters,
        e.g. an exponential distribution truncated to a support
        much shorter than its scale or far from the origin.
        """
        return self.__mean

    @property
    def standard_deviation(self) -> float:
        """The standard deviation of the random variable.

        It is computed from the closed-form expressions
        of the maximum-entropy distributions
        instead of being requested from the library,
        whose moment routines lose their accuracy for some parameters,
        e.g. an exponential distribution truncated to a support
        much shorter than its scale or far from the origin.
        """
        return self.__standard_deviation

    def _substitute_extreme_truncated_normal(
        self,
        family: MaximumEntropyFamily,
        parameters: StrKeyMapping,
        settings: BaseMaximumEntropyDistributionSettings,
    ) -> tuple[MaximumEntropyFamily, StrKeyMapping]:
        """Substitute a truncated normal distribution the library represents poorly.

        Near the largest standard deviation,
        the exact truncated normal distribution has its mode
        far beyond the support in standardized units
        and the library computes its moments inaccurately, or not at all.
        The moments the library would deliver are measured
        and compared with those of the distribution
        maximizing the entropy for the support and the mean only,
        which matches the mean up to the tilt tolerance
        and whose standard deviation is the largest one,
        hence close to the requested one in this regime.
        A warning states the substitution
        and the delivered standard deviation
        when the substitute is used.
        When the truncated normal distribution is kept,
        a warning states its moment error
        only if that error exceeds the accuracy that the fit guarantees.

        Args:
            family: The family of the distribution maximizing the entropy.
            parameters: The parameters of this distribution.
            settings: The information about the random variable.

        Returns:
            The possibly substituted family and parameters.
        """
        if family != MaximumEntropyFamily.TRUNCATED_NORMAL:
            return family, parameters

        numerical_settings = settings.numerical_settings
        substitute = compute_maximum_entropy_parameters(
            parameters[MaximumEntropyParameter.MINIMUM],
            parameters[MaximumEntropyParameter.MAXIMUM],
            settings.mu,
            None,
            settings=numerical_settings,
        )
        substitute_mean, largest_sigma = compute_maximum_entropy_moments(
            *substitute, settings=numerical_settings
        )
        # The substituted distribution has the largest standard deviation,
        # which is close to the requested one in this regime,
        # and matches the mean up to the tilt tolerance,
        # while the library delivers the moments it can actually compute;
        # the measured errors decide,
        # unless the library does not implement the substituted distribution,
        # in which case the truncated normal one is kept,
        # so that validated settings always construct.
        substitution_error = max(
            abs(largest_sigma - settings.sigma), abs(substitute_mean - settings.mu)
        )
        representation_error = self._compute_truncated_normal_representation_error(
            parameters, settings
        )
        error_bound = settings.sigma * numerical_settings.solver_tolerance**0.5
        if representation_error <= substitution_error or not self._is_family_available(
            *substitute
        ):
            if representation_error > error_bound:
                logger.warning(
                    "The truncated normal distribution maximizing the entropy "
                    "for this information is delivered by the library "
                    "with a moment error of about %s.",
                    representation_error,
                )
            return family, parameters

        logger.warning(
            "The truncated normal distribution maximizing the entropy "
            "for this information cannot be delivered accurately by the library; "
            "the %s distribution maximizing the entropy "
            "for the support and the mean only is used instead, "
            "with the standard deviation %s instead of %s.",
            substitute[0],
            largest_sigma,
            settings.sigma,
        )
        return substitute

    @staticmethod
    def _is_family_available(
        family: MaximumEntropyFamily,
        parameters: StrKeyMapping,
    ) -> bool:
        """Check whether the library implements a maximum-entropy family.

        Args:
            family: The family of the distribution.
            parameters: The parameters of the distribution.

        Returns:
            Whether the library implements this distribution.
        """
        return True

    @abstractmethod
    def _compute_truncated_normal_representation_error(
        self,
        parameters: StrKeyMapping,
        settings: BaseMaximumEntropyDistributionSettings,
    ) -> float:
        """Measure the moment error of the truncated normal delivered by the library.

        Args:
            parameters: The parameters of the truncated normal distribution.
            settings: The information about the random variable.

        Returns:
            The largest absolute error of the delivered mean and standard deviation,
            infinite when the library cannot compute them.
        """

    # TODO: remove this method once the OpenTURNS bug has been fixed.
    # See the issue: https://github.com/openturns/openturns/issues/3274.
    def _set_support(self, settings: BaseMaximumEntropyDistributionSettings) -> None:
        """Set the mathematical support from the information about the random variable.

        The mathematical support is exactly
        the interval defined by the minimum and the maximum,
        unbounded on the sides where they are unknown.
        The library may report a narrower support,
        e.g. OpenTURNS estimates the support of a transformed distribution numerically
        and reports it as bounded.

        Args:
            settings: The information about the random variable.
        """
        self.math_lower_bound = -inf if settings.minimum is None else settings.minimum
        self.math_upper_bound = inf if settings.maximum is None else settings.maximum

    @abstractmethod
    def _create_maximum_entropy_settings(
        self,
        family: MaximumEntropyFamily,
        parameters: StrKeyMapping,
        settings: BaseMaximumEntropyDistributionSettings,
    ) -> BaseGenericDistributionSettings:
        """Create the settings of the library-specific distribution.

        Args:
            family: The family of the distribution maximizing the entropy.
            parameters: The parameters of this distribution.
            settings: The information about the random variable.

        Returns:
            The settings of the library-specific distribution.
        """

    @staticmethod
    def _create_standard_parameters(
        family: MaximumEntropyFamily,
        settings: BaseMaximumEntropyDistributionSettings,
    ) -> dict[str, str | int | float]:
        """Create the parameters used for the string representation.

        The string representation uses the information given by the user,
        which is more meaningful than the parameters of the resulting distribution,
        plus the family of that distribution,
        as the class name does not tell which distribution was selected.

        Args:
            family: The family of the distribution maximizing the entropy.
            settings: The information about the random variable.

        Returns:
            The parameters used for the string representation.
        """
        standard_parameters = {_distribution: str(family)}
        # The numerical settings are not part of the information about the variable.
        for name in _information_names:
            value = getattr(settings, name)
            if value is not None:
                standard_parameters[name] = value

        return standard_parameters
