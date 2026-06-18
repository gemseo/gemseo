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
"""Event variable."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.uncertainty.reliability.event import Event
from gemseo.uncertainty.reliability.event import _ElementaryEvent

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.core.functions.array_function import ArrayFunction


class EventVariable:
    """A variable of interest whose comparisons to any threshold produce events.

    E.g. `EventVariable(f) < 3` is an
    [Event][gemseo.uncertainty.reliability.event.Event] associated to `f < 3`.

    An event variable wraps either an
    [ArrayFunction][gemseo.core.functions.array_function.ArrayFunction],
    from which both the name and the function are taken,
    or a name string,
    in which case the function is left unset
    (to be resolved when the event is added to a
    [ReliabilityScenario][gemseo.uncertainty.reliability.scenario.ReliabilityScenario]).
    """

    __name: str
    """The name of the variable of interest."""

    __function: ArrayFunction | None
    """The function evaluating the variable of interest, if known."""

    def __init__(self, function_or_name: ArrayFunction | str) -> None:
        """
        Args:
            function_or_name: Either the function evaluating the variable of interest
                or the name of the variable of interest.
        """  # noqa: D205, D212
        if isinstance(function_or_name, str):
            self.__name = function_or_name
            self.__function = None
        else:
            self.__name = function_or_name.name
            self.__function = function_or_name

    def __create_event(self, threshold: float, greater: bool) -> Event:
        """Create an event from a threshold and a comparison direction.

        Args:
            threshold: The threshold of the elementary event.
            greater: Whether the variable of interest is greater than the threshold.

        Returns:
            The event definined by a single elementary event.
        """
        return Event(
            _ElementaryEvent(
                name=self.__name,
                threshold=threshold,
                greater=greater,
                function=self.__function,
            )
        )

    def __lt__(self, threshold: float) -> Event:
        return self.__create_event(threshold, greater=False)

    def __le__(self, threshold: float) -> Event:
        return self.__create_event(threshold, greater=False)

    def __gt__(self, threshold: float) -> Event:
        return self.__create_event(threshold, greater=True)

    def __ge__(self, threshold: float) -> Event:
        return self.__create_event(threshold, greater=True)

    def isin(self, interval: Sequence[float]) -> Event:
        """Create an event for membership in a continuous interval.

        Args:
            interval: The lower and upper bounds [a, b] of the interval,
                i.e. a <= variable <= b.

        Returns:
            The event defined as a <= variable <= b.
        """
        return self.__create_event(interval[0], greater=True) & self.__create_event(
            interval[1], greater=False
        )

    @classmethod
    def __from_functions_or_names(
        cls, *functions_or_names: ArrayFunction | str
    ) -> EventVariable | tuple[EventVariable, ...]:
        """Create event variables from a list of functions or names.

        Args:
            *functions_or_names: Either the functions
                evaluating the variables of interest
                or the names of the variables of interest.

        Returns:
            The event variables.
        """
        obj = tuple(cls(function_or_name) for function_or_name in functions_or_names)
        if len(obj) == 1:
            return obj[0]

        return obj

    @classmethod
    def from_functions(
        cls, *functions: ArrayFunction
    ) -> EventVariable | tuple[EventVariable, ...]:
        """Create event variables from a list of functions.

        Args:
            *functions: The functions evaluating the variables of interest.

        Returns:
            The event variables.
        """
        return cls.__from_functions_or_names(*functions)

    @classmethod
    def from_names(cls, *names: str) -> EventVariable | tuple[EventVariable, ...]:
        """Create event variables from a list of names.

        Args:
            *names: The names of the variables of interest.

        Returns:
            The event variables.
        """
        return cls.__from_functions_or_names(*names)
