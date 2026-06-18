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
from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.core.functions.array_function import ArrayFunction
from gemseo.uncertainty.reliability.event_variable import EventVariable as V

if TYPE_CHECKING:
    from gemseo.uncertainty.reliability.event import Event


def make_event_comparable(
    event: Event,
) -> list[list[tuple[str, float, bool]]]:
    """Represent an event as a comparable event of (name, threshold, greater).

    This event is expressed in disjonctive normal form (DNF),
    .e. union of intersections.

    Args:
        event: The event.

    Returns:
        The union of sorted intersections.
    """
    return [
        sorted((e.name, e.threshold, e.greater) for e in intersection)
        for intersection in event
    ]


def test_less_than():
    """V(name) < threshold yields a single less-than elementary event."""
    assert make_event_comparable(V("a") < 3) == [[("a", 3, False)]]


def test_greater_than():
    """V(name) > threshold yields a single greater-than elementary event."""
    assert make_event_comparable(V("a") > 3) == [[("a", 3, True)]]


def test_less_equal():
    """<= maps to greater=False like <."""
    assert make_event_comparable(V("a") <= 3) == [[("a", 3, False)]]


def test_greater_equal():
    """>= maps to greater=True like >."""
    assert make_event_comparable(V("a") >= 3) == [[("a", 3, True)]]


def test_reflected_comparison():
    """threshold < V(name) yields a greater-than event via reflection."""
    assert make_event_comparable(2 < V("a")) == [[("a", 2, True)]]  # noqa: SIM300


def test_variable_from_function():
    """A variable built from a function takes its name and function."""
    function = ArrayFunction(sum, name="f")
    event = V(function) < 3
    elementary_event = event[0][0]
    assert elementary_event.name == "f"
    assert elementary_event.function is function


def test_variable_from_name_has_no_function():
    """A variable built from a name has no function."""
    assert (V("a") < 3)[0][0].function is None


def test_and():
    """& builds a single intersection of elementary events."""
    assert make_event_comparable((V("a") < 3) & (V("b") > 4)) == [
        [("a", 3, False), ("b", 4, True)]
    ]


def test_or():
    """| concatenates intersections."""
    assert make_event_comparable((V("a") < 3) | (V("b") > 4)) == [
        [("a", 3, False)],
        [("b", 4, True)],
    ]


def test_and_distributes_over_or():
    """& distributes over | (disjunctive normal form)."""
    assert make_event_comparable(((V("a") < 1) | (V("b") < 2)) & (V("c") < 3)) == [
        [("a", 1, False), ("c", 3, False)],
        [("b", 2, False), ("c", 3, False)],
    ]


def test_isin_interval():
    """isin([a, b]) yields a single intersection of a >= a and a <= b."""
    assert make_event_comparable(V("a").isin([2, 3])) == [
        [("a", 2.0, True), ("a", 3.0, False)]
    ]


def test_isin_str():
    """isin([a, b]) string representation reads as 'x > a AND x < b'."""
    assert str(V("a").isin([2, 3])) == "a > 2.0 AND a < 3.0"


def test_full_event():
    """The target expression yields the expected DNF and string representation."""
    expression = (V("f") < 3) & (V("g") > 4) | (V("h") > 2) & (V("h") < 5)
    assert make_event_comparable(expression) == [
        [("f", 3, False), ("g", 4, True)],
        [("h", 2, True), ("h", 5, False)],
    ]
    assert str(expression) == "(f < 3.0 AND g > 4.0) OR (h > 2.0 AND h < 5.0)"


def test_from_functions_or_names():
    """from_functions_or_names returns event variables."""
    a = V.from_names("a")
    assert a._EventVariable__name == "a"

    a, b, c = V.from_names("a", "b", "c")
    assert a._EventVariable__name == "a"
    assert b._EventVariable__name == "b"
    assert c._EventVariable__name == "c"
