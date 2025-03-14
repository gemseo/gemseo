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
"""Fitting a probability distribution to data using the OpenTURNS library."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Final

import openturns as ots
from openturns import DistributionFactory
from openturns import FittingTest
from openturns import Sample
from openturns import TestResult
from strenum import StrEnum

from gemseo.uncertainty.distributions.base_distribution_fitter import (
    BaseDistributionFitter,
)
from gemseo.uncertainty.distributions.openturns.distribution import OTDistribution

if TYPE_CHECKING:
    from gemseo.typing import RealArray
    from gemseo.typing import StrKeyMapping


def _get_ot_distribution_factories() -> dict[str, type[DistributionFactory]]:
    """Return the OpenTURNS distribution factories.

    Returns:
        The mapping from the distributions to their factories.
    """
    distribution_names_to_ot_factories = {}
    for ot_factory in DistributionFactory.GetContinuousUniVariateFactories():
        if "SmoothedUniformFactory" not in str(ot_factory):
            factory_class_name = ot_factory.getImplementation().getClassName()
            distribution_name = factory_class_name.split("Factory")[0]
            distribution_names_to_ot_factories[distribution_name] = getattr(
                ots, factory_class_name
            )

    return distribution_names_to_ot_factories


class OTDistributionFitter(BaseDistributionFitter[OTDistribution]):
    """Fit a probability distribution to data using the OpenTURNS library."""

    variable: str
    """The name of the variable."""

    _OT_DISTRIBUTION_NAMES_TO_OT_FACTORIES: Final[
        dict[str, type[DistributionFactory]]
    ] = _get_ot_distribution_factories()

    DistributionName: ClassVar[StrEnum] = StrEnum(
        "DistributionName", sorted(_OT_DISTRIBUTION_NAMES_TO_OT_FACTORIES.keys())
    )

    FittingCriterion: ClassVar[StrEnum] = StrEnum(
        "FittingCriterion", "BIC ChiSquared Kolmogorov"
    )

    _CRITERIA_TO_WRAPPED_OBJECTS: ClassVar[dict[FittingCriterion, FittingTest]] = {
        FittingCriterion.BIC: FittingTest.BIC,
        FittingCriterion.ChiSquared: FittingTest.ChiSquared,
        FittingCriterion.Kolmogorov: FittingTest.Kolmogorov,
    }

    SignificanceTest: ClassVar[StrEnum] = StrEnum(
        "SignificanceTest", "ChiSquared Kolmogorov"
    )

    _FITTING_CRITERIA_TO_MINIMIZE: ClassVar[set[FittingCriterion]] = {
        FittingCriterion.BIC
    }

    def __init__(
        self,
        # TODO: API: rename to variable_name or remove it because useless.
        variable: str,
        data: RealArray,
    ) -> None:
        """
        Args:
            variable: The name of the variable.
        """  # noqa: D205,D212,D415
        self.variable = variable
        super().__init__(data)

    @BaseDistributionFitter.data.setter
    def data(self, data_: RealArray) -> None:  # noqa: D102
        self._data = data_
        self._samples = Sample(data_.reshape((-1, 1)))

    def fit(  # noqa: D102
        self,
        distribution: DistributionName,
    ) -> OTDistribution:
        ot_factory = self._OT_DISTRIBUTION_NAMES_TO_OT_FACTORIES[distribution]
        fitted_distribution = ot_factory().build(self._samples)
        return OTDistribution(distribution, fitted_distribution.getParameter())

    def _compute_measure(
        self,
        distribution: OTDistribution | DistributionName,
        criterion: FittingCriterion,
        level: float,
    ) -> Any:
        if not isinstance(distribution, OTDistribution):
            distribution = self.fit(distribution)

        openturns_distribution = distribution.distribution
        openturns_test = self._CRITERIA_TO_WRAPPED_OBJECTS[criterion]
        if criterion in {t.value for t in self.SignificanceTest}:
            return openturns_test(self._samples, openturns_distribution, level)

        return openturns_test(self._samples, openturns_distribution)

    @staticmethod
    def _format_significance_test_goodness_of_fit(
        result: TestResult, level: float
    ) -> tuple[bool, StrKeyMapping]:
        return result.getBinaryQualityMeasure(), {
            "p-value": result.getPValue(),
            "statistics": result.getStatistic(),
            "level": level,
        }
