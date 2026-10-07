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

import operator
from typing import TYPE_CHECKING

import pytest
from numpy import array
from numpy import nan
from numpy.testing import assert_array_equal
from pydantic import ValidationError

from gemseo.core.function.array_function import ArrayFunction
from gemseo.uncertainty.reliability.event import Event
from gemseo.uncertainty.reliability.event_variable import EventVariable as V
from gemseo.uncertainty.reliability.threshold_comparator import ThresholdComparator
from gemseo.util.testing.helper import assert_exception

if TYPE_CHECKING:
    from collections.abc import Callable


def make_event_comparable(
    event: Event,
) -> list[list[tuple[str, ThresholdComparator, float]]]:
    """Represent an event as a comparable event of (name, comparator, threshold).

    This event is expressed in disjunctive normal form (DNF),
    i.e. union of intersections.

    Args:
        event: The event.

    Returns:
        The union of sorted intersections.
    """
    return [
        sorted((e.name, e.comparator, e.threshold) for e in intersection)
        for intersection in event
    ]


def _event_with_conflicting_functions() -> Event:
    """Return an event whose repeated variable name has two different functions.

    Returns:
        An intersection of two elementary events on the same variable name,
        each bound to a distinct function.
    """
    function_1 = ArrayFunction(sum, name="h")
    function_2 = ArrayFunction(sum, name="h")
    return (V(function_1) < 1) & (V(function_2) > 2)


def _event_with_unbound_and_conflicting_functions() -> Event:
    """Return an event with an unbound variable and a conflicting one.

    Returns:
        An intersection of a variable without function
        and of two elementary events on the same variable name,
        each bound to a distinct function.
    """
    return (V("a") < 0) & _event_with_conflicting_functions()


@pytest.mark.parametrize(
    (
        "event",
        "comparator",
        "expected_str",
        "expected_indicator",
        "expected_negated_str",
    ),
    [
        (
            V("a") < 3,
            ThresholdComparator.LESS,
            "a < 3.0",
            [1.0, 0.0, 0.0],
            "a >= 3.0",
        ),
        (
            V("a") > 3,
            ThresholdComparator.GREATER,
            "a > 3.0",
            [0.0, 0.0, 1.0],
            "a <= 3.0",
        ),
        (
            V("a") <= 3,
            ThresholdComparator.LESS_EQUAL,
            "a <= 3.0",
            [1.0, 1.0, 0.0],
            "a > 3.0",
        ),
        (
            V("a") >= 3,
            ThresholdComparator.GREATER_EQUAL,
            "a >= 3.0",
            [0.0, 1.0, 1.0],
            "a < 3.0",
        ),
        (
            3 < V("a"),  # noqa: SIM300
            ThresholdComparator.GREATER,
            "a > 3.0",
            [0.0, 0.0, 1.0],
            "a <= 3.0",
        ),
    ],
    ids=["less", "greater", "less_equal", "greater_equal", "reflected_greater"],
)
def test_comparison(
    event: Event,
    comparator: ThresholdComparator,
    expected_str: str,
    expected_indicator: list[float],
    expected_negated_str: str,
):
    """A comparison operator yields an elementary event that evaluates and negates.

    The sample includes the threshold value itself,
    so strict and non-strict comparisons are told apart.
    """
    assert make_event_comparable(event) == [[("a", comparator, 3.0)]]
    assert str(event) == expected_str
    indicator = event.evaluate({"a": array([2.0, 3.0, 4.0])})
    assert_array_equal(indicator, array(expected_indicator))
    assert str(~event) == expected_negated_str


def test_variable_from_function():
    """A variable built from a function takes its name and function."""
    function = ArrayFunction(sum, name="f")
    event = V(function) < 3
    elementary_event = event[0][0]
    assert elementary_event.name == "f"
    assert elementary_event.function is function


def test_variable_from_name():
    """A variable built from a name has no function."""
    assert (V("a") < 3)[0][0].function is None


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        (
            (V("a") < 3) & (V("b") > 4),
            [
                [
                    ("a", ThresholdComparator.LESS, 3),
                    ("b", ThresholdComparator.GREATER, 4),
                ]
            ],
        ),
        (
            (V("a") < 3) | (V("b") > 4),
            [
                [("a", ThresholdComparator.LESS, 3)],
                [("b", ThresholdComparator.GREATER, 4)],
            ],
        ),
        (
            ((V("a") < 1) | (V("b") < 2)) & (V("c") < 3),
            [
                [
                    ("a", ThresholdComparator.LESS, 1),
                    ("c", ThresholdComparator.LESS, 3),
                ],
                [
                    ("b", ThresholdComparator.LESS, 2),
                    ("c", ThresholdComparator.LESS, 3),
                ],
            ],
        ),
        (
            (V("x") < 1) & (V("x") < 1),
            [[("x", ThresholdComparator.LESS, 1)]],
        ),
        (
            (V("x") < 1) | (V("x") < 1),
            [[("x", ThresholdComparator.LESS, 1)]],
        ),
    ],
    ids=[
        "and",
        "or",
        "and_distributes_over_or",
        "and_deduplicates_repeated_elementary_event",
        "or_deduplicates_repeated_intersection",
    ],
)
def test_combination(
    event: Event, expected: list[list[tuple[str, ThresholdComparator, float]]]
):
    """& builds an intersection, | concatenates, and & distributes over | (DNF).

    & and | also deduplicate repeated elementary events and intersections.
    """
    assert make_event_comparable(event) == expected


def test_or_keeps_first_occurrence_of_duplicate_intersections():
    """| keeps the first written order of intersections that differ only by order."""
    event = (V("a") < 1) & (V("b") > 2) | (V("b") > 2) & (V("a") < 1)
    assert str(event) == "a < 1.0 AND b > 2.0"


def test_isin_interval():
    """isin([a, b]) yields a closed interval, both as DNF and as a string."""
    event = V("a").isin([2, 3])
    assert make_event_comparable(event) == [
        [
            ("a", ThresholdComparator.LESS_EQUAL, 3.0),
            ("a", ThresholdComparator.GREATER_EQUAL, 2.0),
        ]
    ]
    assert str(event) == "a >= 2.0 AND a <= 3.0"


def test_full_event():
    """The target expression yields the expected DNF and string representation."""
    expression = (V("f") < 3) & (V("g") > 4) | (V("h") > 2) & (V("h") < 5)
    assert make_event_comparable(expression) == [
        [("f", ThresholdComparator.LESS, 3), ("g", ThresholdComparator.GREATER, 4)],
        [("h", ThresholdComparator.LESS, 5), ("h", ThresholdComparator.GREATER, 2)],
    ]
    assert str(expression) == "(f < 3.0 AND g > 4.0) OR (h > 2.0 AND h < 5.0)"


@pytest.mark.parametrize(
    ("event", "data", "expected"),
    [
        (
            (V("a") > 0.0) & (V("b") < 1.0),
            {"a": array([1.0, 1.0, -1.0]), "b": array([0.0, 2.0, 0.0])},
            [1.0, 0.0, 0.0],
        ),
        (
            (V("a") > 0.0) | (V("b") > 0.0),
            {"a": array([1.0, -1.0, -1.0]), "b": array([-1.0, 1.0, -1.0])},
            [1.0, 1.0, 0.0],
        ),
    ],
    ids=["intersection", "union"],
)
def test_evaluate_combination(
    event: Event, data: dict[str, object], expected: list[float]
):
    """evaluate combines an intersection with AND, and a union with OR."""
    assert_array_equal(event.evaluate(data), array(expected))


@pytest.mark.parametrize("from_functions", [False, True])
def test_from_functions_or_names(from_functions: bool):
    """from_functions and from_names return event variables bound to their inputs."""
    names = "abc"
    functions = [ArrayFunction(sum, name=name) for name in names]
    create = V.from_functions if from_functions else V.from_names
    inputs = functions if from_functions else list(names)

    a = create(inputs[0])
    # A single input gives a variable, not a tuple.
    assert isinstance(a, V)

    variables = create(*inputs)
    for variable, name, function in zip(variables, names, functions, strict=True):
        elementary_event = (variable < 1)[0][0]
        assert elementary_event.name == name
        assert elementary_event.function is (function if from_functions else None)


@pytest.mark.parametrize(
    "build",
    [
        lambda: 2 < V("h") < 5,
        lambda: (V("f") < 3) and (V("g") > 4),
        lambda: (V("f") < 3) or (V("g") > 4),
        lambda: not (V("f") < 3),
        lambda: bool(V("f") < 3),
    ],
    ids=[
        "chained_comparison",
        "and",
        "or",
        "not",
        "bool",
    ],
)
def test_bool_raises(build: Callable[[], object], snapshot):
    """An Event cannot be used in a boolean context, e.g. via and, or, not or bool."""
    with assert_exception(TypeError, snapshot):
        build()


def test_negation_of_combination():
    """~ applies De Morgan's laws to a combination, both as a string and on evaluate.

    A NaN value satisfies neither the event nor its negation,
    so both evaluate to 0 on the corresponding sample.
    """
    event = (V("a") < 1) & (V("b") > 2) | (V("c") >= 3)
    assert str(~event) == "(a >= 1.0 AND c < 3.0) OR (b <= 2.0 AND c < 3.0)"

    # The last sample has a = NaN;
    # comparisons on a NaN value are always False,
    # so the corresponding elementary event AND its own negation are both False,
    # e.g. a < 1 and its complement a >= 1 are both False for a = NaN.
    # b and c are chosen so that this makes the whole event, and its negation,
    # evaluate to 0 rather than depending on the other clause.
    data = {
        "a": array([0.0, 1.0, 2.0, nan]),
        "b": array([1.0, 2.0, 3.0, 3.0]),
        "c": array([2.0, 3.0, 4.0, 1.0]),
    }
    indicator = event.evaluate(data)
    negated_indicator = (~event).evaluate(data)
    assert_array_equal(negated_indicator[:-1], 1.0 - indicator[:-1])
    assert indicator[-1] == 0.0
    assert negated_indicator[-1] == 0.0


def test_negation_deduplicates_repeated_elementary_events():
    """~ removes elementary events duplicated by the De Morgan distribution."""
    event = ~(((V("x") < 1) & (V("y") < 1)) | ((V("x") < 1) & (V("z") < 1)))
    assert str(event) == (
        "(x >= 1.0) OR (x >= 1.0 AND z >= 1.0) "
        "OR (y >= 1.0 AND x >= 1.0) OR (y >= 1.0 AND z >= 1.0)"
    )


@pytest.mark.parametrize(
    ("build", "needs_small_bound"),
    [
        (lambda: Event().evaluate({}), False),
        (lambda: ~Event(), False),
        (lambda: ~((V("a") < 0) & (V("b") < 0) & (V("c") < 0)), True),
        (lambda: (V("a") < 0).get_functions(), False),
        (lambda: _event_with_conflicting_functions().get_functions(), False),
        (
            lambda: _event_with_unbound_and_conflicting_functions().get_functions(),
            False,
        ),
        (lambda: (V("a") < 1)[0][0].bind_function("not a function"), False),
    ],
    ids=[
        "empty_event_evaluate",
        "empty_event_negation",
        "bound_exceeded",
        "unbound_function",
        "conflicting_functions",
        "unbound_and_conflicting_functions",
        "bind_non_function",
    ],
)
def test_value_error(
    build: Callable[[], object],
    needs_small_bound: bool,
    monkeypatch: pytest.MonkeyPatch,
    snapshot,
):
    """evaluate, ~, get_functions and bind_function raise a ValueError."""
    if needs_small_bound:
        # A patched, small bound keeps the negated event tiny and the test fast:
        # negating a single intersection of 3 elementary events
        # produces a union of 3 intersections,
        # which already exceeds a bound of 2.
        monkeypatch.setattr(Event, "max_intersections", 2)
    with assert_exception(ValueError, snapshot):
        build()


@pytest.mark.parametrize("combine", [operator.and_, operator.or_], ids=["and", "or"])
def test_and_or_type_error_for_non_event(combine: Callable[[object, object], object]):
    """& and | raise a TypeError when the other operand is not an Event."""
    event = V("a") < 3
    with pytest.raises(TypeError):
        combine(event, 42)


def test_map_returns_a_new_immutable_event():
    """map returns a new event, leaving the original, immutable event unmodified."""
    function_a = ArrayFunction(sum, name="a")
    function_b = ArrayFunction(sum, name="b")
    name_to_function = {"a": function_a, "b": function_b}
    event = (V("a") < 1) & (V("b") > 2)

    mapped = event.map(lambda e: e.bind_function(name_to_function[e.name]))

    assert [e.function for intersection in mapped for e in intersection] == [
        function_a,
        function_b,
    ]
    assert [e.function for intersection in event for e in intersection] == [
        None,
        None,
    ]
    with pytest.raises(ValidationError):
        mapped[0][0].threshold = 2.0


@pytest.mark.parametrize(
    ("event_1", "event_2"),
    [
        (400 < V("h"), V("h") > 400),  # noqa: SIM300
        ((V("a") < 1) & (V("b") > 2), (V("b") > 2) & (V("a") < 1)),
        ((V("a") < 1) | (V("b") > 2), (V("b") > 2) | (V("a") < 1)),
    ],
    ids=["reflected", "and_commutes", "or_commutes"],
)
def test_equality(event_1: Event, event_2: Event):
    """Two events describing the same DNF, built differently, compare equal.

    Their hashes also match,
    as required for objects that compare equal.
    """
    assert event_1 == event_2
    assert hash(event_1) == hash(event_2)


@pytest.mark.parametrize(
    "other",
    [
        V("b") < 1,
        V("a") <= 1,
        V("a") < 2,
        "a < 1.0",
    ],
    ids=[
        "different_name",
        "different_comparator",
        "different_threshold",
        "non_event",
    ],
)
def test_inequality(other: object):
    """Events differing by name, comparator or threshold are not equal.

    Nor is an event equal to a non-Event,
    even one whose string happens to match the event's own text,
    which keeps the `NotImplemented` branch of `Event.__eq__` covered.
    """
    assert (V("a") < 1) != other


def test_repr_matches_str():
    """repr mirrors str, so a bare event renders readably, e.g. in a notebook."""
    event = V("h") > 400
    assert repr(event) == str(event) == "h > 400.0"


@pytest.mark.parametrize(
    ("event", "length"),
    [
        (Event(), 0),
        (V("a") < 1, 1),
        (((V("a") < 1) & (V("b") > 2)) | (V("c") >= 3), 2),
    ],
    ids=["empty_event", "elementary_event", "union_of_two_intersections"],
)
def test_len(event: Event, length: int):
    """The length of an event is its number of intersections."""
    assert len(event) == length
