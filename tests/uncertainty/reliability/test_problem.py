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

from gemseo.algos.parameter_space import ParameterSpace
from gemseo.core.functions.array_function import ArrayFunction
from gemseo.uncertainty.distributions.openturns.normal_settings import (
    OTNormalDistribution_Settings,
)
from gemseo.uncertainty.reliability.problem import ReliabilityProblem
from gemseo.utils.testing.helpers import assert_exception


@pytest.fixture(scope="module")
def uncertain_space() -> ParameterSpace:
    """The uncertain space."""
    space = ParameterSpace()
    space.add_random_variable("u", OTNormalDistribution_Settings())
    return space


def test_problem(uncertain_space):
    """Test ReliabilityProblem."""
    function_1 = ArrayFunction(sum, name="f1")
    function_2 = ArrayFunction(sum, name="f2")

    problem = ReliabilityProblem(uncertain_space)
    f1, f2 = problem.get_event_variables(function_1, function_2)
    problem.add_event(f1 > 0, event_name="a")
    problem.add_event((f2 > 0) & (f1 > 0))

    assert list(problem.name_to_event.keys()) == ["a", "event_2"]
    assert list(problem.observables) == [function_1, function_2]


def test_event(uncertain_space):
    """An Event is stored directly when added."""
    function_1 = ArrayFunction(sum, name="f1")
    function_2 = ArrayFunction(sum, name="f2")

    problem = ReliabilityProblem(uncertain_space)
    f1, f2 = problem.get_event_variables(function_1, function_2)
    problem.add_event((f1 < 3) & (f2 > 4), event_name="a")

    (event_1, event_2) = problem.name_to_event["a"][0]
    assert (event_1.name, event_1.threshold, event_1.greater, event_1.function) == (
        "f1",
        3,
        False,
        function_1,
    )
    assert (event_2.name, event_2.threshold, event_2.greater, event_2.function) == (
        "f2",
        4,
        True,
        function_2,
    )
    assert list(problem.observables) == [function_1, function_2]


def test_event_without_function(uncertain_space, snapshot):
    """Test ReliabilityProblem raises when function field is None."""
    problem = ReliabilityProblem(uncertain_space)
    f = problem.get_event_variables("f")
    with assert_exception(ValueError, snapshot):
        problem.add_event(f > 0, event_name="a")


def test_string_representation(uncertain_space):
    """Test ReliabilityProblem._get_string_representation."""
    function_1 = ArrayFunction(sum, name="f1")
    function_2 = ArrayFunction(sum, name="f2")

    problem = ReliabilityProblem(uncertain_space)
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
