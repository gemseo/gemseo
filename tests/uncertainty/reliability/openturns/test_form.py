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
from openturns import FORMResult

from gemseo.core.functions.array_function import ArrayFunction
from gemseo.uncertainty.reliability.openturns.form import OT_FORM
from gemseo.uncertainty.reliability.openturns.form_settings import OT_FORM_Settings
from gemseo.uncertainty.reliability.openturns.optimizer import BaseOTOptimizer
from gemseo.uncertainty.reliability.problem import ReliabilityProblem
from gemseo.utils.testing.helpers import assert_exception


@pytest.mark.parametrize("settings", [None, *BaseOTOptimizer.__subclasses__()])
@pytest.mark.parametrize(
    ("greater", "expected"), [(None, 0.25), (True, 0.25), (False, 0.75)]
)
def test_form(function, uncertain_space, settings, greater, expected):
    """Test OT_FORM."""
    form = OT_FORM()
    kwargs = {}
    if settings is not None:
        kwargs["settings"] = OT_FORM_Settings(optimizer=settings())

    problem = ReliabilityProblem(uncertain_space)
    f = problem.get_event_variables(function)
    problem.add_event(
        f < 0.75 if greater is False else f > 0.75,
        event_name="a",
    )
    results = form.execute(problem, **kwargs)
    assert len(results) == 1
    assert results["a"].name == "a"
    probability = results["a"].probability
    raw_result = results["a"].raw_result
    assert probability == pytest.approx(expected, abs=1e-3)
    assert isinstance(raw_result, FORMResult)
    assert raw_result.getEventProbability() == probability


def test_type_errors(uncertain_space, f, snapshot):
    """Check the type errors when events are wrong."""
    problem = ReliabilityProblem(uncertain_space)
    form = OT_FORM()
    with assert_exception(ValueError, snapshot):
        form.execute(problem)

    problem = ReliabilityProblem(uncertain_space)
    function = ArrayFunction(f, name="y")
    f = problem.get_event_variables(function)
    problem.add_event((f > 0.75) & (f > 0.75), event_name="a")
    with assert_exception(TypeError, snapshot):
        form.execute(problem)


@pytest.mark.parametrize(
    ("use_database", "expected"), [(False, 0), (True, 29), (None, 0)]
)
def test_database(problem, use_database, expected, enable_function_statistics):
    """Test the use of the database."""
    kwargs = {} if use_database is None else {"use_database": use_database}
    form = OT_FORM()
    form.execute(problem, settings=OT_FORM_Settings(**kwargs))
    assert len(problem.database) == expected
    assert problem.observables[0].n_calls == 32
