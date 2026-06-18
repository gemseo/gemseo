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
"""Directional sampling algorithm."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from openturns import DirectionalSampling
from openturns import Null
from openturns import OrthogonalDirection
from openturns import RandomDirection

from gemseo.uncertainty.reliability.openturns.base import BaseOTReliabilityAlgorithm
from gemseo.uncertainty.reliability.openturns.directional_sampling_settings import (
    OT_DirectionalSampling_Settings,
)
from gemseo.uncertainty.reliability.result import ReliabilityResult

if TYPE_CHECKING:
    from gemseo.uncertainty.reliability.problem import ReliabilityProblem


class OT_DirectionalSampling(BaseOTReliabilityAlgorithm):  # noqa: N801
    """The directional sampling algorithm."""

    settings_class: ClassVar[type[OT_DirectionalSampling_Settings]] = (
        OT_DirectionalSampling_Settings
    )

    _ALGO_CLASS: ClassVar[type[DirectionalSampling]] = DirectionalSampling

    def _execute(
        self,
        event_name: str,
        problem: ReliabilityProblem,
        settings: OT_DirectionalSampling_Settings,
    ) -> ReliabilityResult:
        self._set_seed(settings.seed)

        solver_settings = settings.root_strategy.solver
        solver = solver_settings.ALGO_CLASS()
        solver.setAbsoluteError(solver_settings.absolute_error)
        solver.setRelativeError(solver_settings.relative_error)
        solver.setResidualError(solver_settings.residual_error)
        solver.setMaximumCallsNumber(solver_settings.maximum_calls_number)

        root_strategy = settings.root_strategy.ALGO_CLASS()
        root_strategy.setSolver(solver)
        root_strategy.setStepSize(settings.root_strategy.step_size)
        root_strategy.setMaximumDistance(settings.root_strategy.maximum_distance)

        ot_event = self._create_ot_event(event_name, problem)

        algo = self._ALGO_CLASS(ot_event)
        algo.setConvergenceStrategy(Null())
        algo.setRootStrategy(root_strategy)
        algo.setSamplingStrategy(
            RandomDirection()
            if settings.use_random_sampling_strategy
            else OrthogonalDirection()
        )
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
