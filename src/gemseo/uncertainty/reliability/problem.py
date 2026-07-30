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
"""Reliability analysis problem."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.core.problem.evaluation import EvaluationProblem
from gemseo.uncertainty.reliability.event import Event
from gemseo.uncertainty.reliability.event_variable import EventVariable
from gemseo.util.string import MultiLineString

if TYPE_CHECKING:
    from gemseo.core.function.array_function import ArrayFunction
    from gemseo.space.parameter import ParameterSpace


class ReliabilityProblem(EvaluationProblem):
    """A reliability analysis problem."""

    __name_to_event: dict[str, Event]
    """The map from an event name to an event."""

    design_space: ParameterSpace

    def __init__(  # noqa: D107
        self,
        design_space: ParameterSpace,
        differentiation_method: EvaluationProblem.DifferentiationMethod = EvaluationProblem.DifferentiationMethod.USER,  # noqa: E501
        differentiation_step: float = 1e-7,
        parallel_differentiation: bool = False,
        **parallel_differentiation_options: int | bool,
    ) -> None:
        super().__init__(
            design_space,
            differentiation_method=differentiation_method,
            differentiation_step=differentiation_step,
            parallel_differentiation=parallel_differentiation,
            **parallel_differentiation_options,
        )
        self.__name_to_event = {}

    def add_event(self, event: Event, event_name: str = "") -> None:
        """Add an event.

        Args:
            event: The event
                built from variables and boolean and comparison operators,
                e.g. `(f < 3) & (g > 4) | (2 < h) & (h < 5)`
                where the variables are created using
                [get_event_variables][gemseo.uncertainty.reliability.problem.ReliabilityProblem.get_event_variables]
                as `f, g, h = scenario.get_event_variables(func_f, func_g, func_h)`.
            event_name: The name to be given to this event.
                If empty, use `"event_i"` for the i-th event.

        Raises:
            ValueError: If a function field of the events is `None`.
        """
        if not event_name:
            event_name = f"{Event.default_name}_{len(self.__name_to_event) + 1}"

        observables = []
        for intersection_event in event:
            for elementary_event in intersection_event:
                function = elementary_event.function
                if function not in self.observables:
                    if function is None:
                        msg = (
                            "The function field of the elementary event "
                            f"{elementary_event.name!r} cannot be None."
                        )
                        raise ValueError(msg)

                    observables.append(function)

        for observable in observables:
            self.add_observable(observable)

        self.__name_to_event[event_name] = event

    def _get_string_representation(self) -> MultiLineString:
        mls = MultiLineString()
        mls.add("Reliability analysis problem:")
        mls.indent()
        mls.add("Compute the probabilities of the events:")
        mls.indent()
        for union_name, union_event in self.__name_to_event.items():
            mls.add("{}: {}", union_name, union_event)
        return mls

    @property
    def name_to_event(self) -> dict[str, Event]:
        """The map from an event name to an event."""
        return self.__name_to_event

    @staticmethod
    def get_event_variables(
        *functions: ArrayFunction,
    ) -> EventVariable | tuple[EventVariable, ...]:
        """Return event variables.

        Args:
            *functions: The functions evaluating the variables of interest.

        Returns:
            The event variables.
        """
        return EventVariable.from_functions(*functions)
