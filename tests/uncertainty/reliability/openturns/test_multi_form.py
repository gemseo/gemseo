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
from numpy import array
from openturns import MultiFORMResult

from gemseo.algos.parameter_space import ParameterSpace
from gemseo.core.functions.array_function import ArrayFunction
from gemseo.uncertainty.distributions.openturns.normal_settings import (
    OTNormalDistribution_Settings,
)
from gemseo.uncertainty.reliability.openturns.multi_form import OT_MultiFORM
from gemseo.uncertainty.reliability.openturns.multi_form_settings import (
    OT_MultiFORM_Settings,
)
from gemseo.uncertainty.reliability.openturns.optimizer import BaseOTOptimizer
from gemseo.uncertainty.reliability.problem import ReliabilityProblem


def f(x):
    """An offset function."""
    return array([5.0 - x[1] - 0.5 * (x[0] - 0.1) ** 2])


@pytest.fixture(scope="module")
def function() -> ArrayFunction:
    """An offset ArrayFunction."""
    return ArrayFunction(f, name="z")


@pytest.fixture(scope="module")
def uncertain_space() -> ParameterSpace:
    """The uncertainty space."""
    parameter_space = ParameterSpace()
    parameter_space.add_random_vector(
        "u", OTNormalDistribution_Settings(), OTNormalDistribution_Settings()
    )
    return parameter_space


@pytest.mark.parametrize(
    "settings", [None, *(cls for cls in BaseOTOptimizer.__subclasses__())]
)
@pytest.mark.parametrize(
    ("greater", "expected"),
    [
        (None, (0.0028, 0.9982, 0.9990)),
        (True, (0.0028, 0.9982, 0.9990)),
        (False, (0.0028, 0.0018, 0.0010)),
    ],
)
def test_multi_form(function, uncertain_space, settings, greater, expected):
    """Test OT_MultiFORM."""
    multi_form = OT_MultiFORM()
    kwargs = {}
    if settings is not None:
        kwargs["settings"] = OT_MultiFORM_Settings(optimizer=settings())

    problem = ReliabilityProblem(uncertain_space)
    f = problem.get_event_variables(function)
    problem.add_event(
        f < 0.0 if greater is False else f > 0.0,
        event_name="a",
    )
    results = multi_form.execute(problem, **kwargs)
    assert len(results) == 1
    assert results["a"].name == "a"
    probability = results["a"].probability
    raw_result = results["a"].raw_result
    assert probability == pytest.approx(expected[0], abs=1e-4)
    assert isinstance(raw_result, MultiFORMResult)
    assert raw_result.getEventProbability() == probability
    form_results = raw_result.getFORMResultCollection()
    assert len(form_results) == 2
    probabilities = sorted([
        form_results[0].getEventProbability(),
        form_results[1].getEventProbability(),
    ])
    expected = sorted(expected[1:])
    assert probabilities[0] == pytest.approx(expected[0], abs=1e-4)
    assert probabilities[1] == pytest.approx(expected[1], abs=1e-4)
