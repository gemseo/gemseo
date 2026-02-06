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
"""The interface to SciPy-based probability distributions."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

import scipy.stats as scipy_stats
from scipy.stats._distn_infrastructure import rv_continuous_frozen

from gemseo.typing import StrKeyMapping
from gemseo.uncertainty.distributions.base_distribution import BaseDistribution
from gemseo.uncertainty.distributions.scalar_distribution_mixin import (
    ScalarDistributionMixin,
)
from gemseo.uncertainty.distributions.scipy.distribution_settings import (
    SPDistribution_Settings,
)
from gemseo.uncertainty.distributions.scipy.joint import SPJointDistribution

if TYPE_CHECKING:
    from collections.abc import Generator

    from numpy.random import RandomState

    from gemseo.typing import RealArray


class SPDistribution(
    BaseDistribution[float, StrKeyMapping, rv_continuous_frozen],
    ScalarDistributionMixin,
):
    """A SciPy-based probability distribution.

    Warning:
       The distribution parameters must be provided according to the signature
       of the scipy classes. [Access the scipy documentation](https://docs.scipy.org/doc/scipy/reference/stats.html).
    """

    settings_class = SPDistribution_Settings

    JOINT_DISTRIBUTION_CLASS: ClassVar[type[SPJointDistribution]] = SPJointDistribution

    _WEBSITE: ClassVar[str] = "https://docs.scipy.org/doc/scipy/reference/stats.html"

    def _create_distribution(self, settings: SPDistribution_Settings) -> None:
        distribution = self._create_distribution_from_module(scipy_stats, settings)
        self.math_lower_bound, self.math_upper_bound = distribution.interval(1.0)
        extrema_level = 1e-12
        self.num_lower_bound = distribution.ppf(extrema_level)
        self.num_upper_bound = distribution.ppf(1 - extrema_level)
        self.distribution = distribution

    def compute_samples(  # noqa: D102
        self,
        n_samples: int = 1,
        random_state: int | Generator | RandomState | None = None,
    ) -> RealArray:
        """
        Args:
            random_state: The SciPy random state.
        """  # noqa: D205, D212
        return self.distribution.rvs(n_samples, random_state)

    def compute_cdf(  # noqa: D102
        self,
        value: float,
    ) -> float:
        return self.distribution.cdf(value)

    def compute_inverse_cdf(  # noqa: D102
        self,
        value: float,
    ) -> float:
        return self.distribution.ppf(value)

    @property
    def mean(self) -> float:  # noqa: D102
        return self.distribution.mean()

    @property
    def standard_deviation(self) -> float:  # noqa: D102
        return self.distribution.std()

    def _pdf(
        self,
        value: float,
    ) -> float:
        return self.distribution.pdf(value)

    def _cdf(
        self,
        level: float,
    ) -> float:
        return self.distribution.cdf(level)
