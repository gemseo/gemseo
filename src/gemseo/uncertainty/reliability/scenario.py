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
"""Reliability analysis scenario."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING
from typing import ClassVar

from gemseo.scenarios.evaluation import EvaluationScenario
from gemseo.uncertainty.reliability import EventVariable
from gemseo.uncertainty.reliability.factory import ReliabilityAlgorithmFactory
from gemseo.uncertainty.reliability.problem import ReliabilityProblem

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.algos.parameter_space import ParameterSpace
    from gemseo.core.discipline.base_discipline import BaseDiscipline
    from gemseo.core.functions.function_from_discipline import FunctionFromDiscipline
    from gemseo.formulations.base_settings import BaseFormulationSettings
    from gemseo.uncertainty.reliability.event import Event
    from gemseo.uncertainty.reliability.result import ReliabilityResult


class ReliabilityScenario(EvaluationScenario):
    """A reliability analysis scenario."""

    _ALGO_FACTORY_CLASS: ClassVar[type[ReliabilityAlgorithmFactory]] = (
        ReliabilityAlgorithmFactory
    )

    __name_to_function: dict[str, FunctionFromDiscipline]
    """The map from a disciplinary output name to a function."""

    __problem: ReliabilityProblem
    """The reliability analysis problem."""

    def __init__(  # noqa: D107
        self,
        disciplines: Sequence[BaseDiscipline],
        design_space: ParameterSpace,
        name: str = "",
        formulation_settings: BaseFormulationSettings | None = None,
    ) -> None:
        super().__init__(
            disciplines,
            design_space,
            name=name,
            formulation_settings=formulation_settings,
        )
        self.__name_to_function = {}
        self.__problem = ReliabilityProblem(design_space)
        self._execution_result = {}

    def add_event(self, event: Event, event_name: str = "") -> None:
        """Add an event.

        Args:
            event: The event
                built from variables and boolean and comparison operators,
                e.g. `(f < 3) & (g > 4) | (2 < h) & (h < 5)`
                where the variables are created using
                [get_event_variables][gemseo.uncertainty.reliability.scenario.ReliabilityScenario.get_event_variables]
                as `f, g, h = scenario.get_event_variables("f", "g", "h")`.
            event_name: The name to be given to this event.
                If empty, use `"event_i"` for the i-th event.
        """
        processed_event = deepcopy(event)
        for intersection_event in processed_event:
            for elementary_event in intersection_event:
                output_name = elementary_event.name
                function = self.__name_to_function.get(output_name)
                if function is None:
                    function = self.formulation.create_function((output_name,))
                    self.__name_to_function[output_name] = function

                elementary_event.function = function

        self.__problem.add_event(processed_event, event_name=event_name)

    def _execute(self) -> None:
        settings = self._algorithm_settings
        algo = self._algo_factory.create(settings.target_class_name)
        self._execution_result = algo.execute(self.__problem, settings=settings)

    @property
    def event_name_to_reliability_result(self) -> dict[str, ReliabilityResult]:
        """The map from an event name to a reliability analysis result."""
        return self._execution_result

    @staticmethod
    def get_event_variables(*names: str) -> EventVariable | tuple[EventVariable, ...]:
        """Return event variables.

        Args:
            *names: The names of the event variables.

        Returns:
            The event variables.
        """
        return EventVariable.from_names(*names)
