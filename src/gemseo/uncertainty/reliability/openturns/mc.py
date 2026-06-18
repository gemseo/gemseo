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
"""Monte Carlo sampling algorithm."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from openturns import Null
from openturns import ProbabilitySimulationAlgorithm

from gemseo.uncertainty.reliability.openturns.base import BaseOTReliabilityAlgorithm
from gemseo.uncertainty.reliability.openturns.mc_settings import OT_MC_Settings
from gemseo.uncertainty.reliability.result import ReliabilityResult

if TYPE_CHECKING:
    from gemseo.uncertainty.reliability.problem import ReliabilityProblem


class OT_MC(BaseOTReliabilityAlgorithm):  # noqa: N801
    """The Monte Carlo sampling algorithm."""

    settings_class: ClassVar[type[OT_MC_Settings]] = OT_MC_Settings

    _ALGO_CLASS: ClassVar[type[ProbabilitySimulationAlgorithm]] = (
        ProbabilitySimulationAlgorithm
    )

    def _execute(
        self,
        event_name: str,
        problem: ReliabilityProblem,
        settings: OT_MC_Settings,
    ) -> ReliabilityResult:
        self._set_seed(settings.seed)

        ot_event = self._create_ot_event(event_name, problem)

        algo = self._ALGO_CLASS(ot_event, settings.create_experiment())
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
