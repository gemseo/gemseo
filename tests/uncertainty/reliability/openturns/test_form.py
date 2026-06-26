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
from numpy.testing import assert_almost_equal
from openturns import AnalyticalResult
from openturns import FORMResult

from gemseo.core.functions.array_function import ArrayFunction
from gemseo.uncertainty.reliability.openturns.form import OT_FORM
from gemseo.uncertainty.reliability.openturns.form_settings import OT_FORM_Settings
from gemseo.uncertainty.reliability.openturns.optimizer import BaseOTOptimizer
from gemseo.uncertainty.reliability.openturns.optimizer import NLoptAlgorithmName
from gemseo.uncertainty.reliability.openturns.optimizer import OTNLopt
from gemseo.uncertainty.reliability.problem import ReliabilityProblem
from gemseo.utils.comparisons import compare_dict_of_arrays
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
    reliability_index = results["a"].reliability_index
    raw_result = results["a"].raw_result
    assert probability == pytest.approx(expected, abs=1e-3)
    assert isinstance(raw_result, FORMResult)
    assert raw_result.getEventProbability() == probability
    assert raw_result.getHasoferReliabilityIndex() == reliability_index

    design_point = results["a"].design_point
    assert_almost_equal(design_point.physical, raw_result.getPhysicalSpaceDesignPoint())
    assert_almost_equal(design_point.standard, raw_result.getStandardSpaceDesignPoint())
    assert compare_dict_of_arrays(
        design_point.physical_as_dict,
        uncertain_space.convert_array_to_dict(design_point.physical),
    )
    assert compare_dict_of_arrays(
        design_point.standard_as_dict,
        uncertain_space.convert_array_to_dict(design_point.standard),
    )

    importance_factors = results["a"].importance_factors
    assert_almost_equal(
        importance_factors.classical,
        raw_result.getImportanceFactors(AnalyticalResult.CLASSICAL),
    )
    assert_almost_equal(
        importance_factors.elliptical,
        raw_result.getImportanceFactors(AnalyticalResult.ELLIPTICAL),
    )
    assert_almost_equal(
        importance_factors.physical,
        raw_result.getImportanceFactors(AnalyticalResult.PHYSICAL),
    )
    assert compare_dict_of_arrays(
        importance_factors.classical_as_dict,
        uncertain_space.convert_array_to_dict(importance_factors.classical),
    )
    assert compare_dict_of_arrays(
        importance_factors.elliptical_as_dict,
        uncertain_space.convert_array_to_dict(importance_factors.elliptical),
    )
    assert compare_dict_of_arrays(
        importance_factors.physical_as_dict,
        uncertain_space.convert_array_to_dict(importance_factors.physical),
    )


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
    ("use_database", "length", "n_calls"),
    [(False, (0, 0), (36, 32)), (True, (31, 29), (36, 32)), (None, (0, 0), (36, 32))],
)
def test_database(problem, use_database, length, n_calls, enable_function_statistics):
    """Test the use of the database."""
    index = int(problem.observables[0].has_jac)
    kwargs = {} if use_database is None else {"use_database": use_database}
    form = OT_FORM()
    form.execute(problem, settings=OT_FORM_Settings(**kwargs))
    assert len(problem.database) == length[index]
    assert problem.observables[0].n_calls == n_calls[index]


def test_form_slsqp(parametric_f_and_j, uncertain_space):
    """Test OT_FORM using a gradient-based optimizer."""
    f, j = parametric_f_and_j
    function = ArrayFunction(f, name="y", jac=j)

    problem = ReliabilityProblem(uncertain_space)
    f = problem.get_event_variables(function)
    problem.add_event(f < 0.75, event_name="a")

    form = OT_FORM()
    results = form.execute(
        problem,
        OT_FORM_Settings(optimizer=OTNLopt(algo_name=NLoptAlgorithmName.LD_SLSQP)),
    )
    assert results["a"].probability == pytest.approx(0.75, abs=1e-3)
