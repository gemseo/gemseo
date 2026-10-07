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
from gemseo.space.random import RandomSpace
from gemseo.uncertainty.reliability.event import Event
from gemseo.uncertainty.reliability.event_variable import EventVariable
from gemseo.util.string import MultiLineString
from gemseo.util.string import pretty_repr

if TYPE_CHECKING:
    from gemseo.core.function.array_function import ArrayFunction


class ReliabilityProblem(EvaluationProblem[RandomSpace]):
    """A reliability analysis problem."""

    __name_to_event: dict[str, Event]
    """The map from an event name to an event."""

    def __init__(  # noqa: D107
        self,
        random_space: RandomSpace,
        differentiation_method: EvaluationProblem.DifferentiationMethod = EvaluationProblem.DifferentiationMethod.USER,  # noqa: E501
        differentiation_step: float = 1e-7,
        parallel_differentiation: bool = False,
        **parallel_differentiation_options: int | bool,
    ) -> None:
        super().__init__(
            random_space,
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
                as `f, g, h = problem.get_event_variables(func_f, func_g, func_h)`.
            event_name: The name to be given to this event.
                If empty, use `"event_i"` for the i-th event.

        Raises:
            ValueError: If the event contains no elementary event,
                if a function field of the events is `None`,
                or if two elementary events sharing the same variable name
                are bound to two different functions.
        """
        if len(event) == 0:
            msg = (
                "The event must be an Event instantiated "
                "from at least one ElementaryEvent."
            )
            raise ValueError(msg)

        if not event_name:
            event_name = f"{Event.default_name}_{len(self.__name_to_event) + 1}"

        functions = event.get_functions()
        for function in functions:
            if function not in self.observables:
                self.add_observable(function)

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

        Raises:
            ValueError: If two of the functions share the same name.
        """
        names = [function.name for function in functions]
        duplicate_names = sorted({name for name in names if names.count(name) > 1})
        if duplicate_names:
            label = "name" if len(duplicate_names) == 1 else "names"
            msg = (
                "The functions must have different names; "
                f"several of them share the {label} {pretty_repr(duplicate_names)}."
            )
            raise ValueError(msg)

        return EventVariable.from_functions(*functions)
