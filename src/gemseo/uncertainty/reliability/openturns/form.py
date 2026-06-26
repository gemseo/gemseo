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

from numpy import array
from openturns import FORM
from openturns import AbdoRackwitz
from openturns import AnalyticalResult
from openturns import Cobyla
from openturns import NearestPointProblem
from openturns import NLopt
from openturns import OptimizationAlgorithmImplementation
from openturns import Point

from gemseo.uncertainty.reliability.openturns.base import BaseOTReliabilityAlgorithm
from gemseo.uncertainty.reliability.openturns.form_result import MPFP
from gemseo.uncertainty.reliability.openturns.form_result import FORMResult
from gemseo.uncertainty.reliability.openturns.form_result import ImportanceFactors
from gemseo.uncertainty.reliability.openturns.form_settings import OT_FORM_Settings
from gemseo.uncertainty.reliability.openturns.multi_form_result import MultiFORMResult
from gemseo.uncertainty.reliability.openturns.optimizer import BaseOTOptimizer

if TYPE_CHECKING:
    from openturns import Analytical
    from openturns import FORMResult as OTFORMResult

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

    _USE_MULTIFORM_RESULT: ClassVar[bool] = False
    """Whether the algorithm returns a `MultiFORMResult`."""

    def _execute(
        self,
        event_name: str,
        problem: ReliabilityProblem,
        settings: OT_FORM_Settings,
    ) -> FORMResult | MultiFORMResult:
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
        if self._USE_MULTIFORM_RESULT:
            return MultiFORMResult(
                name=event_name,
                probability=self._extract_probability(result, settings),
                raw_result=result,
                reliability_index=result.getGeneralisedReliabilityIndex(),
                form_results=tuple(
                    self.__create_form_result(
                        event_name, problem, form_result, settings
                    )
                    for form_result in result.getFORMResultCollection()
                ),
            )

        return self.__create_form_result(event_name, problem, result, settings)

    def __create_form_result(
        self,
        event_name: str,
        problem: ReliabilityProblem,
        result: OTFORMResult,
        settings: OT_FORM_Settings,
    ) -> FORMResult:
        """Create a FORMResult from an OpenTURNS FORMResult.

        Args:
            event_name: The name of the event.
            problem: The reliability problem.
            result: The OpenTURNS FORMResult.
            settings: The settings of the reliability algorithm.

        Returns:
            The FORMResult.
        """
        physical_mpfp = array(result.getPhysicalSpaceDesignPoint())
        standard_mpfp = array(result.getStandardSpaceDesignPoint())
        convert_array_to_dict = problem.design_space.convert_array_to_dict
        design_point = MPFP(
            physical=physical_mpfp,
            standard=standard_mpfp,
            physical_as_dict=convert_array_to_dict(physical_mpfp),
            standard_as_dict=convert_array_to_dict(standard_mpfp),
        )
        classical = array(result.getImportanceFactors(AnalyticalResult.CLASSICAL))
        elliptical = array(result.getImportanceFactors(AnalyticalResult.ELLIPTICAL))
        physical = array(result.getImportanceFactors(AnalyticalResult.PHYSICAL))
        importance_factors = ImportanceFactors(
            classical=classical,
            classical_as_dict=convert_array_to_dict(classical),
            elliptical=elliptical,
            elliptical_as_dict=convert_array_to_dict(elliptical),
            physical=physical,
            physical_as_dict=convert_array_to_dict(physical),
        )
        return FORMResult(
            design_point=design_point,
            importance_factors=importance_factors,
            name=event_name,
            probability=self._extract_probability(result, settings),
            raw_result=result,
            reliability_index=result.getHasoferReliabilityIndex(),
        )

    @staticmethod
    def _set_algo_options(algo: Analytical, settings: OT_FORM_Settings) -> None:
        """Set the option of the OpenTURNS algorithm.

        Args:
            algo: The OpenTURNS algorithm.
        """

    @staticmethod
    def _extract_probability(result: OTFORMResult, settings: OT_FORM_Settings) -> float:
        """Get the probability from an OpenTURNS result.

        Args:
            result: The OpenTURNS result.
            settings: The settings of the reliability algorithm.

        Returns:
            The probability.
        """
        return result.getEventProbability()
