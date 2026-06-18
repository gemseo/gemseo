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
"""Base class for importance sampling algorithms."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from openturns import Null

from gemseo.uncertainty.reliability.openturns.base import BaseOTReliabilityAlgorithm
from gemseo.uncertainty.reliability.result import ReliabilityResult

if TYPE_CHECKING:
    from gemseo.uncertainty.reliability.openturns.base_is_settings import (
        BaseOTISSettings,
    )
    from gemseo.uncertainty.reliability.problem import ReliabilityProblem


class BaseOTImportanceSampling(BaseOTReliabilityAlgorithm):  # noqa: N801
    """The base class for importance sampling algorithms."""

    settings_class: ClassVar[type[BaseOTISSettings]]

    def _execute(
        self,
        event_name: str,
        problem: ReliabilityProblem,
        settings: BaseOTISSettings,
    ) -> ReliabilityResult:
        self._set_seed(settings.seed)

        ot_event = self._create_ot_event(event_name, problem)
        args = (getattr(settings, name) for name in settings.INSTANTIATION_ARGUMENTS)
        algo = self._ALGO_CLASS(ot_event, *args)
        algo.setConvergenceStrategy(Null())
        algo.setMaximumCoefficientOfVariation(settings.maximum_coefficient_of_variation)
        algo.setMaximumOuterSampling(settings.maximum_outer_sampling)
        algo.setMaximumStandardDeviation(settings.maximum_standard_deviation)
        algo.run()

        result = algo.getResult()
        return ReliabilityResult(
            name=event_name,
            probability=result.getProbabilityEstimate(),
            raw_result=result,
        )
