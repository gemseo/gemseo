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
"""Event."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import ClassVar

if TYPE_CHECKING:
    from collections.abc import Iterator

    from gemseo.core.functions.array_function import ArrayFunction


@dataclass
class _ElementaryEvent:
    """An elementary event defined by a single threshold comparison."""

    name: str
    """The name of the variable of interest."""

    threshold: float = 0.0
    """The threshold of the probability."""

    greater: bool = True
    """Whether the variable of interest is greater than the threshold."""

    function: ArrayFunction | None = None
    """The function evaluating the variable of interest, if known."""

    def __post_init__(self) -> None:
        self.threshold = float(self.threshold)

    def __str__(self) -> str:
        operator = ">" if self.greater else "<"
        return f"{self.name} {operator} {self.threshold}"


class Event:
    """An event in its disjunctive normal form (DNF), i.e. union of intersections.

    It is defined using a fluent and math-like syntax
    from an [EventVariable][gemseo.uncertainty.reliability.event_variable.EventVariable]
    the comparison operators `<`, `<=`, `>` and `>=`,
    and the boolean operators `&` (AND, a.k.a. intersection) and `|` (OR, a.k.a. union):

    ```python
    event = (EventVariable(f) < 3) & (EventVariable(g) > 4) | (
        2 < EventVariable(h)
    ) & (EventVariable(h) < 5)
    ```

    reads as `((f < 3) AND (g > 4)) OR ((h > 2) AND (h < 5))`.

    !!! warning
        Each elementary comparison must be parenthesized,
        e.g. `(EventVariable(f) < 3) & (EventVariable(g) > 4)` and not `EventVariable(f) < 3 & EventVariable(g) > 4`.
        Chained comparisons such as `2 < EventVariable(h) < 5` are not supported.
        Write `(2 < EventVariable(h)) & (EventVariable(h) < 5)` instead.
    """  # noqa: E501

    default_name: ClassVar[str] = "event"
    """The default name of an event."""

    __intersections: list[tuple[_ElementaryEvent, ...]]
    """The intersections of elementary events."""

    def __init__(self, *events: _ElementaryEvent) -> None:
        """
        Args:
            *events: The elementary events of a single intersection.
        """  # noqa: D205, D212
        self.__intersections = [tuple(events)] if events else []

    def __and__(self, other: Event) -> Event:
        # DNF distribution: (a1|a2) & (b1|b2) = a1b1 | a1b2 | a2b1 | a2b2.
        result = Event()
        result.__intersections = [
            a + b for a in self.__intersections for b in other.__intersections
        ]
        return result

    def __or__(self, other: Event) -> Event:
        result = Event()
        result.__intersections = self.__intersections + other.__intersections
        return result

    def __iter__(self) -> Iterator[tuple[_ElementaryEvent, ...]]:
        return iter(self.__intersections)

    def __getitem__(self, index: int) -> tuple[_ElementaryEvent, ...]:
        return self.__intersections[index]

    def __str__(self) -> str:
        if len(self.__intersections) > 1:
            return " OR ".join(
                f"({' AND '.join(str(e) for e in intersection)})"
                for intersection in self.__intersections
            )
        parts = self.__intersections[0] if self.__intersections else ()
        return " AND ".join(str(e) for e in parts)

    @property
    def is_combination(self) -> bool:
        """Whether the event is a combination of elementary events."""
        return len(self.__intersections) > 1 or (
            len(self.__intersections) == 1 and len(self.__intersections[0]) > 1
        )
