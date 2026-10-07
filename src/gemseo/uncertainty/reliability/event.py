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

import operator
from functools import reduce
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import NoReturn
from typing import TypeVar

from gemseo.util.string import pretty_repr

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable
    from collections.abc import Iterator
    from collections.abc import Mapping

    from gemseo.core.function.array_function import ArrayFunction
    from gemseo.uncertainty.reliability.elementary_event import ElementaryEvent
    from gemseo.util.typing import BooleanArray
    from gemseo.util.typing import RealArray

_T = TypeVar("_T")
"""The type returned by folding an event."""


class Event:
    """An event in its disjunctive normal form (DNF), i.e. union of intersections.

    It is defined using a fluent and math-like syntax
    from an [EventVariable][gemseo.uncertainty.reliability.event_variable.EventVariable]
    the comparison operators `<`, `<=`, `>` and `>=`,
    and the boolean operators `&` (AND, a.k.a. intersection),
    `|` (OR, a.k.a. union)
    and `~` (NOT, a.k.a. negation):

    ```python
    event = (EventVariable(f) < 3) & (EventVariable(g) > 4) | (
        2 < EventVariable(h)
    ) & (EventVariable(h) < 5)
    ```

    reads as `((f < 3) AND (g > 4)) OR ((h > 2) AND (h < 5))`.

    Two events compare equal when they have the same set of intersections,
    each intersection being the same set of elementary events,
    independently of the order of the intersections
    and of the elementary events within each of them.

    !!! warning
        Each elementary comparison must be parenthesized,
        e.g. `(EventVariable(f) < 3) & (EventVariable(g) > 4)`
        and not `EventVariable(f) < 3 & EventVariable(g) > 4`:
        Python reads the latter as
        `EventVariable(f) < (3 & EventVariable(g)) > 4`,
        and `3 & EventVariable(g)` raises a `TypeError`.
        Chained comparisons such as `2 < EventVariable(h) < 5`
        and the Python operators `and`, `or` and `not` also raise a `TypeError`,
        because each of them relies on the truth value
        of an intermediate `Event`,
        which has none,
        as an `Event` is a variable rather than a value of one.
        Write `(2 < EventVariable(h)) & (EventVariable(h) < 5)`
        instead of `2 < EventVariable(h) < 5`,
        or `EventVariable(h).isin([2, 5])` for the closed interval `2 <= h <= 5`.
        Write `~(EventVariable(h) > 5)` instead of `not (EventVariable(h) > 5)`.
    """

    default_name: ClassVar[str] = "event"
    """The default name of an event."""

    max_intersections: ClassVar[int] = 1024
    r"""The maximum number of intersections that the negation of an event may produce.

    Negating a union of $N$ intersections of $n_1, \ldots, n_N$ elementary events,
    with De Morgan's laws,
    produces up to $n_1 \times \ldots \times n_N$ intersections
    once expanded back to DNF,
    i.e. $n^N$ when all the intersections have $n$ elementary events,
    which can grow very large.
    """

    __intersections: tuple[tuple[ElementaryEvent, ...], ...]
    """The intersections of elementary events."""

    def __init__(self, *events: ElementaryEvent) -> None:
        """
        Args:
            *events: The elementary events of a single intersection.
        """  # noqa: D205, D212
        self.__intersections = _normalize_intersections(
            (tuple(events),) if events else ()
        )

    @classmethod
    def __from_intersections(
        cls, intersections: Iterable[Iterable[ElementaryEvent]]
    ) -> Event:
        """Create an event from its intersections of elementary events.

        Args:
            intersections: The intersections of elementary events.

        Returns:
            The event,
            with the duplicate elementary events and intersections removed,
            as in
            `_normalize_intersections`.
        """
        event = cls()
        event.__intersections = _normalize_intersections(intersections)
        return event

    def __bool__(self) -> NoReturn:
        msg = (
            "An Event is a variable, not a value of this variable, "
            "so it has no truth value. "
            'Combine events with "&", "|" and "~" '
            'instead of "and", "or" and "not", '
            "parenthesize each comparison, "
            'and write "(a < x) & (x < b)" or "x.isin([a, b])" instead of "a < x < b".'
        )
        raise TypeError(msg)

    def __and__(self, other: Event) -> Event:
        if not isinstance(other, Event):
            return NotImplemented

        # DNF distribution: (a1|a2) & (b1|b2) = a1 & b1 | a1 & b2 | a2 & b1 | a2 & b2.
        return Event.__from_intersections(
            a + b for a in self.__intersections for b in other.__intersections
        )

    def __or__(self, other: Event) -> Event:
        if not isinstance(other, Event):
            return NotImplemented

        return Event.__from_intersections(self.__intersections + other.__intersections)

    def __invert__(self) -> Event:
        r"""Negate the event.

        On an elementary event,
        the negation is exact for non-NaN values
        and returns the complementary comparison;
        a NaN value satisfies neither the elementary event nor its negation.
        On a combination,
        De Morgan's laws are applied
        and the result is re-expanded into disjunctive normal form.

        Negating a combination can be costly:
        a union of $N$ intersections of $n_1, \ldots, n_N$ elementary events
        has a negation of up to $n_1 \times \ldots \times n_N$ intersections,
        i.e. $n^N$ when all the intersections have $n$ elementary events,
        before the duplicates are removed.
        `Event.max_intersections`
        bounds this growth:
        the negation stops with a `ValueError`
        as soon as the number of intersections exceeds it,
        instead of building an event too large to evaluate.

        Returns:
            The negated event.

        Raises:
            ValueError: If the event has no intersections of elementary events,
                or if the negation would produce more than
                `Event.max_intersections` intersections.
        """
        if not self.__intersections:
            msg = (
                "The negation of an event "
                "with no intersections of elementary events is not defined."
            )
            raise ValueError(msg)

        # De Morgan: ~(OR_i AND_j e_ij) = AND_i (OR_j ~e_ij), re-expanded to DNF.
        # __from_intersections, used by & through __and__,
        # already deduplicates elementary events and intersections,
        # so the growing negated_event is normalized incrementally,
        # and the bound below is checked on the deduplicated count.
        negated_event = None
        for intersection in self.__intersections:
            clause = Event.__from_intersections(
                (~elementary_event,) for elementary_event in intersection
            )
            negated_event = clause if negated_event is None else negated_event & clause
            if len(negated_event.__intersections) > self.max_intersections:
                msg = (
                    "The negation of this event would produce more than "
                    f"{self.max_intersections} intersections; "
                    "increase Event.max_intersections "
                    "or rewrite the event to avoid this negation."
                )
                raise ValueError(msg)

        return negated_event

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Event):
            return NotImplemented

        return _as_frozenset(self.__intersections) == _as_frozenset(
            other.__intersections
        )

    def __hash__(self) -> int:
        return hash(_as_frozenset(self.__intersections))

    def __iter__(self) -> Iterator[tuple[ElementaryEvent, ...]]:
        return iter(self.__intersections)

    def __len__(self) -> int:
        """Return the number of intersections of this event.

        An empty event has no intersection,
        so its length is zero.

        Returns:
            The number of intersections.
        """
        return len(self.__intersections)

    def __getitem__(self, index: int) -> tuple[ElementaryEvent, ...]:
        return self.__intersections[index]

    def __str__(self) -> str:
        if len(self.__intersections) > 1:
            return " OR ".join(
                f"({' AND '.join(str(e) for e in intersection)})"
                for intersection in self.__intersections
            )
        parts = self.__intersections[0] if self.__intersections else ()
        return " AND ".join(str(e) for e in parts)

    def __repr__(self) -> str:
        return str(self)

    @property
    def is_combination(self) -> bool:
        """Whether the event is a combination of elementary events."""
        return len(self.__intersections) > 1 or (
            len(self.__intersections) == 1 and len(self.__intersections[0]) > 1
        )

    def get_functions(self) -> tuple[ArrayFunction, ...]:
        """Return the functions evaluating the event's variables of interest.

        Returns:
            The functions, one per variable name, in first-occurrence order.

        Raises:
            ValueError: If some elementary events have no bound function,
                or if some variable names are shared by elementary events
                bound to two different functions.
                All the errors are gathered in a single message.
        """
        name_to_function: dict[str, ArrayFunction] = {}
        unbound: dict[str, None] = {}
        conflicting: dict[str, None] = {}
        for intersection in self.__intersections:
            for elementary_event in intersection:
                name = elementary_event.name
                function = elementary_event.function
                if function is None:
                    unbound[name] = None
                elif name_to_function.setdefault(name, function) is not function:
                    conflicting[name] = None

        messages = []
        if unbound:
            messages.append(
                f"The variables {pretty_repr(list(unbound))} are bound to no function."
            )

        if conflicting:
            messages.append(
                f"The variables {pretty_repr(list(conflicting))} "
                "are bound to two different functions."
            )

        if messages:
            raise ValueError(" ".join(messages))

        return tuple(name_to_function.values())

    def map(self, function: Callable[[ElementaryEvent], ElementaryEvent]) -> Event:
        """Return a new event with each elementary event transformed by a function.

        The callback receives an
        [ElementaryEvent][gemseo.uncertainty.reliability.elementary_event.ElementaryEvent],
        an immutable model with the fields
        `name`, `threshold`, `comparator` and `function`.
        The callback must return an elementary event.
        A new elementary event is obtained
        with `bind_function(function)`
        or with the constructor `ElementaryEvent(...)`,
        whereas `model_copy(update={...})` does not validate the values.
        Mutating the callback's argument in place
        raises a Pydantic `ValidationError`.

        Args:
            function: The function transforming an elementary event
                into another elementary event.

        Returns:
            The new event.
        """
        return Event.__from_intersections(
            (function(e) for e in intersection) for intersection in self.__intersections
        )

    def _fold(
        self,
        elementary: Callable[[ElementaryEvent], _T],
        intersection: Callable[[Iterable[_T]], _T],
        union: Callable[[Iterable[_T]], _T],
    ) -> _T:
        """Fold the event, in disjunctive normal form, into a single value.

        This generalizes the disjunctive normal form
        `OR_i (AND_j elementary(e_ij))`
        by replacing AND and OR with any pair of user functions,
        e.g. boolean combination or backend-specific events.

        `intersection` and `union` each receive an iterator of values,
        rather than a list,
        so that a streaming reduction, e.g. `functools.reduce`,
        never has to hold more than a couple of intermediate values at once,
        which matters for large arrays.

        Args:
            elementary: The function mapping an elementary event to a value.
            intersection: The function combining,
                for a single intersection,
                an iterator of the values of its elementary events
                into a single value.
            union: The function combining an iterator of the values
                of the intersections into a single value.

        Returns:
            The value of the event.

        Notes:
            If the event has no intersections of elementary events,
            `union` is called with an empty iterator,
            which most implementations of `union` cannot handle;
            callers should validate this case beforehand.
        """
        return union(
            intersection(elementary(e) for e in events)
            for events in self.__intersections
        )

    def evaluate(self, name_to_value: Mapping[str, RealArray]) -> RealArray:
        """Simulate the event from data.

        Args:
            name_to_value: The map from a variable name to a variable value.

        Returns:
            The 0/1 indicator of the event, element-wise.

        Raises:
            ValueError: If the event has no intersections of elementary events.
        """
        if not self.__intersections:
            msg = "The event has no intersections of elementary events."
            raise ValueError(msg)

        def evaluate_elementary(elementary_event: ElementaryEvent) -> BooleanArray:
            """Compute the boolean indicator of an elementary event.

            Args:
                elementary_event: The elementary event.

            Returns:
                The boolean indicator of the elementary event.
            """
            return elementary_event.evaluate(name_to_value[elementary_event.name])

        def intersect(indicators: Iterable[BooleanArray]) -> BooleanArray:
            """Combine boolean indicator arrays with an element-wise AND.

            The accumulator is reused and mutated in place,
            to avoid one array allocation per combination,
            which matters for large arrays.
            This is safe only because each indicator is a new array,
            computed from a comparison or a previous in-place combination,
            and shared with no other object.

            Args:
                indicators: The boolean indicator arrays, one per elementary event.

            Returns:
                The element-wise AND of the arrays.
            """
            return reduce(operator.iand, indicators)

        def unite(indicators: Iterable[BooleanArray]) -> BooleanArray:
            """Combine boolean indicator arrays with an element-wise OR.

            The accumulator is reused and mutated in place,
            to avoid one array allocation per combination,
            which matters for large arrays.
            This is safe only because each indicator is a new array,
            computed from an intersection or a previous in-place combination,
            and shared with no other object.

            Args:
                indicators: The boolean indicator arrays, one per intersection.

            Returns:
                The element-wise OR of the arrays.
            """
            return reduce(operator.ior, indicators)

        indicator = self._fold(evaluate_elementary, intersect, unite)
        return indicator.astype(float)


def _as_frozenset(
    intersections: Iterable[Iterable[ElementaryEvent]],
) -> frozenset[frozenset[ElementaryEvent]]:
    """Represent intersections as a frozenset of frozensets.

    This representation is order-independent,
    both across intersections and within each of them,
    and is used for the equality and the hash of an `Event`.

    Args:
        intersections: The intersections of elementary events.

    Returns:
        The frozenset of frozensets of elementary events.
    """
    return frozenset(frozenset(intersection) for intersection in intersections)


def _deduplicate_events(
    events: Iterable[ElementaryEvent],
) -> tuple[ElementaryEvent, ...]:
    """Remove duplicate elementary events from an intersection.

    Two elementary events are considered duplicates here
    when they compare equal,
    i.e. when their name, threshold, comparator and function all match.

    Args:
        events: The elementary events of a single intersection.

    Returns:
        The events, without duplicates, in their original order.
    """
    return tuple(dict.fromkeys(events))


def _deduplicate_intersections(
    intersections: Iterable[tuple[ElementaryEvent, ...]],
) -> tuple[tuple[ElementaryEvent, ...], ...]:
    """Remove duplicate intersections of elementary events.

    Two intersections are considered duplicates here
    only when their elementary events are pairwise duplicates,
    in the sense of
    `_deduplicate_events`,
    regardless of order,
    so that e.g. `~a & ~b` and `~b & ~a` are recognized as duplicates too.

    Args:
        intersections: The intersections of elementary events,
            each already free of duplicate elementary events.

    Returns:
        The intersections, without duplicates, in their original order,
        keeping the first occurrence of each.
    """
    first_occurrences: dict[
        frozenset[ElementaryEvent], tuple[ElementaryEvent, ...]
    ] = {}
    for intersection in intersections:
        first_occurrences.setdefault(frozenset(intersection), intersection)
    return tuple(first_occurrences.values())


def _normalize_intersections(
    intersections: Iterable[Iterable[ElementaryEvent]],
) -> tuple[tuple[ElementaryEvent, ...], ...]:
    """Deduplicate the elementary events within, and across, intersections.

    Args:
        intersections: The intersections of elementary events.

    Returns:
        The intersections,
        with the duplicate elementary events removed within each intersection
        by
        `_deduplicate_events`
        and the duplicate intersections removed across them
        by
        `_deduplicate_intersections`,
        both in their original order.
    """
    return _deduplicate_intersections(
        _deduplicate_events(intersection) for intersection in intersections
    )
