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

import pytest

from gemseo.core.function.array_function import ArrayFunction
from gemseo.space.random import RandomSpace
from gemseo.uncertainty.distribution.openturns.normal_settings import (
    OTNormalDistribution_Settings,
)
from gemseo.uncertainty.reliability.event import Event
from gemseo.uncertainty.reliability.event_variable import EventVariable
from gemseo.uncertainty.reliability.problem import ReliabilityProblem
from gemseo.uncertainty.reliability.threshold_comparator import ThresholdComparator
from gemseo.util.testing.helper import assert_exception


@pytest.fixture(scope="module")
def random_space() -> RandomSpace:
    """The random space."""
    space = RandomSpace()
    space.add_variable("u", OTNormalDistribution_Settings())
    return space


def test_problem(random_space):
    """Test ReliabilityProblem."""
    function_1 = ArrayFunction(sum, name="f1")
    function_2 = ArrayFunction(sum, name="f2")

    problem = ReliabilityProblem(random_space)
    f1, f2 = problem.get_event_variables(function_1, function_2)
    problem.add_event(f1 > 0, event_name="a")
    problem.add_event((f2 > 0) & (f1 > 0))

    assert list(problem.name_to_event.keys()) == ["a", "event_2"]
    assert list(problem.observables) == [function_1, function_2]


def test_problem_from_random_space():
    """Check that a random space is used as is."""
    space = RandomSpace()
    space.add_variable("u", OTNormalDistribution_Settings())
    problem = ReliabilityProblem(space)
    assert problem.input_space is space


def test_event(random_space):
    """An Event is stored directly when added."""
    function_1 = ArrayFunction(sum, name="f1")
    function_2 = ArrayFunction(sum, name="f2")

    problem = ReliabilityProblem(random_space)
    f1, f2 = problem.get_event_variables(function_1, function_2)
    problem.add_event((f1 < 3) & (f2 > 4), event_name="a")

    (event_1, event_2) = problem.name_to_event["a"][0]
    assert (event_1.name, event_1.threshold, event_1.comparator, event_1.function) == (
        "f1",
        3,
        ThresholdComparator.LESS,
        function_1,
    )
    assert (event_2.name, event_2.threshold, event_2.comparator, event_2.function) == (
        "f2",
        4,
        ThresholdComparator.GREATER,
        function_2,
    )
    assert list(problem.observables) == [function_1, function_2]


def _add_event_with_no_function(problem: ReliabilityProblem) -> None:
    """Add an event whose single elementary event has no function.

    Args:
        problem: The reliability analysis problem.
    """
    f = EventVariable("f")
    problem.add_event(f > 0, event_name="a")


def _add_event_with_partial_function(problem: ReliabilityProblem) -> None:
    """Add an event with two elementary events on "f", only one bound to a function.

    Args:
        problem: The reliability analysis problem.
    """
    function = ArrayFunction(sum, name="f")
    unbound_f = EventVariable("f")
    bound_f = problem.get_event_variables(function)
    problem.add_event((unbound_f < 1) & (bound_f > 0), event_name="a")


def _add_empty_event(problem: ReliabilityProblem) -> None:
    """Add an event with no elementary event.

    Args:
        problem: The reliability analysis problem.
    """
    problem.add_event(Event(), event_name="a")


@pytest.mark.parametrize(
    "add_event",
    [
        _add_event_with_no_function,
        _add_event_with_partial_function,
        _add_empty_event,
    ],
    ids=["no_function", "partial_function", "empty_event"],
)
def test_add_event_value_error(random_space, add_event, snapshot):
    """add_event raises on an event with no intersections or an unbound variable.

    This covers an event with no intersections of elementary events,
    an elementary event with no function,
    and an elementary event sharing a variable name
    with another one bound to a function.
    """
    problem = ReliabilityProblem(random_space)
    with assert_exception(ValueError, snapshot):
        add_event(problem)


def test_get_event_variables_duplicate_names(random_space, snapshot):
    """get_event_variables raises when two functions share the same name."""
    problem = ReliabilityProblem(random_space)
    function_1 = ArrayFunction(sum, name="f")
    function_2 = ArrayFunction(sum, name="f")
    with assert_exception(ValueError, snapshot):
        problem.get_event_variables(function_1, function_2)


def test_add_event_isin_deduplicates_observable(random_space, caplog):
    """add_event with isin does not add the same observable twice."""
    function = ArrayFunction(sum, name="h")

    problem = ReliabilityProblem(random_space)
    h = problem.get_event_variables(function)
    problem.add_event(h.isin([2, 5]), event_name="a")

    observable_names = [observable.name for observable in problem.observables]
    assert observable_names.count("h") == 1
    assert 'already observes "h"' not in caplog.text


def test_string_representation(random_space):
    """Test ReliabilityProblem._get_string_representation."""
    function_1 = ArrayFunction(sum, name="f1")
    function_2 = ArrayFunction(sum, name="f2")

    problem = ReliabilityProblem(random_space)
    f1, f2 = problem.get_event_variables(function_1, function_2)

    problem.add_event(f1 > 0, event_name="a")
    problem.add_event((f2 > 0) & (f1 > 0), event_name="b")

    expected = (
        "Reliability analysis problem:\n"
        "   Compute the probabilities of the events:\n"
        "      a: f1 > 0.0\n"
        "      b: f2 > 0.0 AND f1 > 0.0"
    )
    assert repr(problem) == expected
