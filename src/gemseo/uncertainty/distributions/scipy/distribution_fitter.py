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
"""Fitting a probability distribution to data using the SciPy library."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

import scipy.stats as scipy_stats
from scipy.stats import goodness_of_fit
from scipy.stats import rv_continuous
from strenum import StrEnum

from gemseo.uncertainty.distributions.base_distribution_fitter import (
    BaseDistributionFitter,
)
from gemseo.uncertainty.distributions.scipy.distribution import SPDistribution

if TYPE_CHECKING:
    from scipy.stats._fit import GoodnessOfFitResult

    from gemseo.typing import StrKeyMapping


class SPDistributionFitter(BaseDistributionFitter[SPDistribution]):
    """Fit a probability distribution to data using the SciPy library."""

    DistributionName: ClassVar[StrEnum] = StrEnum(
        "DistributionName",
        [rv.__name__.rsplit("_gen")[0] for rv in rv_continuous.__subclasses__()],
    )

    class FittingCriterion(StrEnum):  # noqa: D106
        ANDERSON_DARLING = "AndersonDarling"
        CRAMER_VON_MISES = "CramerVonMises"
        FILLIBEN = "Filliben"
        KOLMOGOROV_SMIRNOV = "KolmogorovSmirnov"

    _CRITERIA_TO_WRAPPED_OBJECTS: ClassVar[dict[FittingCriterion, str]] = {
        FittingCriterion.ANDERSON_DARLING: "ad",
        FittingCriterion.CRAMER_VON_MISES: "cvm",
        FittingCriterion.FILLIBEN: "filliben",
        FittingCriterion.KOLMOGOROV_SMIRNOV: "ks",
    }

    SignificanceTest: ClassVar[FittingCriterion] = FittingCriterion

    def fit(  # noqa: D102
        self,
        distribution: DistributionName,
    ) -> SPDistribution:
        scipy_distribution = getattr(scipy_stats, f"{distribution}")
        parameters = scipy_distribution.fit(self._samples)
        return SPDistribution(distribution, parameters)

    def _compute_measure(
        self,
        distribution: SPDistribution | DistributionName,
        criterion: FittingCriterion,
        level: float,
    ) -> Any:
        if isinstance(distribution, SPDistribution):
            distribution = distribution.distribution.dist.name

        return goodness_of_fit(
            getattr(scipy_stats, distribution),
            self._samples,
            statistic=self._CRITERIA_TO_WRAPPED_OBJECTS[criterion],
            random_state=0,
        )

    @staticmethod
    def _format_significance_test_goodness_of_fit(
        result: GoodnessOfFitResult, level: float
    ) -> tuple[bool, StrKeyMapping]:
        return result.pvalue >= level, {
            "p-value": result.pvalue,
            "statistics": result.statistic,
            "level": level,
        }
