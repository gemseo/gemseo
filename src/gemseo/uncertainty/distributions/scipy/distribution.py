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
from typing import Any
from typing import ClassVar

import scipy.stats as scipy_stats
from scipy.stats._distn_infrastructure import rv_continuous_frozen

from gemseo.typing import StrKeyMapping
from gemseo.uncertainty.distributions.base_distribution import BaseDistribution
from gemseo.uncertainty.distributions.scalar_distribution_mixin import (
    ScalarDistributionMixin,
)
from gemseo.uncertainty.distributions.scipy.joint import SPJointDistribution
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT

if TYPE_CHECKING:
    from collections.abc import Generator

    from numpy.random import RandomState

    from gemseo.typing import RealArray
    from gemseo.uncertainty.distributions.base_distribution import (
        StandardParametersType,
    )
    from gemseo.uncertainty.distributions.base_joint import BaseJointDistribution


class SPDistribution(
    BaseDistribution[float, StrKeyMapping, rv_continuous_frozen],
    ScalarDistributionMixin,
):
    """A SciPy-based probability distribution.

    .. warning::

       The distribution parameters must be provided according to the signature
       of the scipy classes. `Access the scipy documentation
       <https://docs.scipy.org/doc/scipy/reference/stats.html>`_.

    Examples:
        >>> from gemseo.uncertainty.distributions.scipy.distribution import (
        ...     SPDistribution,
        ... )
        >>> distribution = SPDistribution("expon", {"loc": 3, "scale": 1 / 2.0})
        >>> print(distribution)
        expon(loc=3, scale=0.5)
    """

    JOINT_DISTRIBUTION_CLASS: ClassVar[type[BaseJointDistribution]] = (
        SPJointDistribution
    )

    _WEBSITE: ClassVar[str] = "https://docs.scipy.org/doc/scipy/reference/stats.html"

    def __init__(  # noqa: D107
        self,
        interfaced_distribution: str = "uniform",
        parameters: StrKeyMapping | tuple[Any, ...] = READ_ONLY_EMPTY_DICT,
        standard_parameters: StandardParametersType = READ_ONLY_EMPTY_DICT,
    ) -> None:
        """
        Args:
            standard_parameters: The parameters of the probability distribution
                used for string representation only
                (use ``parameters`` for computation).
                If empty, use ``parameters`` instead.
                For instance,
                let us consider the interfaced SciPy distribution ``"uniform"``.
                Then,
                the string representation of
                ``SPDistribution("uniform", parameters, 1, {"min": 1, "max": 3})``
                with ``parameters={"loc": 1, "scale": 2}``
                is ``"uniform(max=3, min=1)"``
                while the string representation of
                ``SPDistribution("uniform", parameters)``
                is ``"uniform(loc=1, scale=2)"``.
        """  # noqa: D205 D212 D415
        super().__init__(interfaced_distribution, parameters, standard_parameters)

    def _create_distribution(
        self,
        distribution_name: str,
        parameters: StrKeyMapping,
    ) -> None:
        distribution = self._create_distribution_from_module(
            scipy_stats, distribution_name, parameters
        )
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
