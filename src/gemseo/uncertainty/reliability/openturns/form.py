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
"""First-order reliability method (FORM)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Final

from openturns import FORM
from openturns import AbdoRackwitz
from openturns import Cobyla
from openturns import NearestPointProblem
from openturns import NLopt
from openturns import OptimizationAlgorithmImplementation
from openturns import Point

from gemseo.uncertainty.reliability.openturns.base import BaseOTReliabilityAlgorithm
from gemseo.uncertainty.reliability.openturns.form_settings import OT_FORM_Settings
from gemseo.uncertainty.reliability.openturns.optimizer import BaseOTOptimizer
from gemseo.uncertainty.reliability.result import ReliabilityResult

if TYPE_CHECKING:
    from openturns import Analytical
    from openturns import FORMResult

    from gemseo.uncertainty.reliability.problem import ReliabilityProblem


class OT_FORM(BaseOTReliabilityAlgorithm):  # noqa: N801
    """The first-order reliability method (FORM)."""

    settings_class: ClassVar[type[OT_FORM_Settings]] = OT_FORM_Settings

    _ALGO_CLASS: ClassVar[type[FORM]] = FORM

    __NAMES_TO_CLASSES: Final[dict[str, type[OptimizationAlgorithmImplementation]]] = {
        "OTAbdoRackwitz": AbdoRackwitz,
        "OTCobyla": Cobyla,
        "OTNLopt": NLopt,
    }
    """The map from the name of an optimization algorithm to its class."""

    def _execute(
        self,
        event_name: str,
        problem: ReliabilityProblem,
        settings: OT_FORM_Settings,
    ) -> ReliabilityResult:
        opt_settings = settings.optimizer
        opt_class = self.__NAMES_TO_CLASSES[opt_settings.__class__.__name__]
        ot_settings = opt_settings.model_dump(exclude=set(BaseOTOptimizer.model_fields))
        opt = opt_class(NearestPointProblem(), *ot_settings.values())
        opt.setMaximumAbsoluteError(opt_settings.maximum_absolute_error)
        opt.setMaximumCallsNumber(opt_settings.maximum_calls_number)
        opt.setMaximumConstraintError(opt_settings.maximum_constraint_error)
        opt.setMaximumIterationNumber(opt_settings.maximum_iteration_number)
        opt.setMaximumRelativeError(opt_settings.maximum_relative_error)
        opt.setMaximumResidualError(opt_settings.maximum_residual_error)
        opt.setMaximumTimeDuration(opt_settings.maximum_time_duration)
        opt.setStartingPoint(Point(problem.design_space.distribution.mean))

        ot_event = self._create_ot_event(event_name, problem)

        algo = self._ALGO_CLASS(opt, ot_event)
        self._set_algo_options(algo, settings)
        algo.run()

        result = algo.getResult()
        return ReliabilityResult(
            name=event_name,
            probability=self._extract_probability(result, settings),
            raw_result=result,
        )

    @staticmethod
    def _set_algo_options(algo: Analytical, settings: OT_FORM_Settings) -> None:
        """Set the option of the OpenTURNS algorithm.

        Args:
            algo: The OpenTURNS algorithm.
        """

    @staticmethod
    def _extract_probability(result: FORMResult, settings: OT_FORM_Settings) -> float:
        """Get the probability from an OpenTURNS result.

        Args:
            result: The OpenTURNS result.
            settings: The settings of the reliability algorithm.

        Returns:
            The probability.
        """
        return result.getEventProbability()
