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

import pytest
from openturns import MultiFORMResult

from gemseo.algos.parameter_space import ParameterSpace
from gemseo.core.functions.array_function import ArrayFunction
from gemseo.uncertainty.distributions.openturns.uniform_settings import (
    OTUniformDistribution_Settings,
)
from gemseo.uncertainty.reliability.openturns.optimizer import BaseOTOptimizer
from gemseo.uncertainty.reliability.openturns.optimizer import OTNLopt
from gemseo.uncertainty.reliability.openturns.system_form import OT_SystemFORM
from gemseo.uncertainty.reliability.openturns.system_form_settings import (
    OT_SystemFORM_Settings,
)
from gemseo.uncertainty.reliability.problem import ReliabilityProblem
from gemseo.utils.testing.helpers import assert_exception

if TYPE_CHECKING:
    from gemseo.typing import RealArray


def f_1(u: RealArray) -> RealArray:
    """The function returning the first input.

    Args:
        u: The input value.

    Returns:
        The output value.
    """
    return u[[0]]


def f_2(u: RealArray) -> RealArray:
    """The function returning the first input.

    Args:
        u: The input value.

    Returns:
        The output value.
    """
    return u[[1]]


@pytest.fixture(scope="module")
def function_1() -> ArrayFunction:
    """The ArrayFunction wrapping f_1."""
    return ArrayFunction(f_1, name="y")


@pytest.fixture(scope="module")
def function_2() -> ArrayFunction:
    """The ArrayFunction wrapping f_2."""
    return ArrayFunction(f_2, name="z")


@pytest.fixture(scope="module")
def uncertain_space() -> ParameterSpace:
    """The uncertainty space defined by the random vector u=(u1,u2), ui ~ U([0,1])."""
    parameter_space = ParameterSpace()
    parameter_space.add_random_vector(
        "u", OTUniformDistribution_Settings(), OTUniformDistribution_Settings()
    )
    return parameter_space


@pytest.mark.parametrize(
    "settings",
    [
        None,
        *(
            cls
            for cls in BaseOTOptimizer.__subclasses__()
            if cls != OTNLopt
            # RuntimeError: Exception :
            # Obtained design point is not on the limit state:
            # its image by the limit state function is 0.749731,
            # which is incompatible with the threshold: 0.75
            # considering the limit state tolerance of the optimization algorithm: 1e-05
        ),
    ],
)
@pytest.mark.parametrize(
    ("greater_1", "greater_2", "expected"),
    [
        (None, None, 0.0625),  # P[y>0.75]P[z>0.75]=0.25²
        (True, None, 0.0625),  # P[y>0.75]P[z>0.75]=0.25²
        (False, None, 0.1875),  # P[y<0.75]P[z>0.75]=0.75*0.25
        (None, True, 0.0625),  # P[y>0.75]P[z>0.75]=0.25²
        (True, True, 0.0625),  # P[y>0.75]P[z>0.75]=0.25²
        (False, True, 0.1875),  # P[y<0.75]P[z>0.75]=0.75*0.25
        (None, False, 0.1875),  # P[y>0.75]P[z<0.75]=0.75*0.25
        (True, False, 0.1875),  # P[y>0.75]P[z<0.75]=0.75*0.25
        (False, False, 0.5625),  # P[y<0.75]P[z<0.75]=0.75*0.75
    ],
)
def test_system_form(
    function_1,
    function_2,
    uncertain_space,
    settings,
    greater_1,
    greater_2,
    expected,
):
    """Test OT_SystemFORM."""
    system_form = OT_SystemFORM()
    kwargs = {}
    if settings is not None:
        kwargs["settings"] = OT_SystemFORM_Settings(optimizer=settings())

    problem = ReliabilityProblem(uncertain_space)
    f1, f2 = problem.get_event_variables(function_1, function_2)
    e1 = f1 < 0.75 if greater_1 is False else f1 > 0.75
    e2 = f2 < 0.75 if greater_2 is False else f2 > 0.75
    problem.add_event(e1 & e2, event_name="a")
    results = system_form.execute(problem, **kwargs)
    assert len(results) == 1
    assert results["a"].name == "a"
    probability = results["a"].probability
    raw_result = results["a"].raw_result
    assert probability == pytest.approx(expected, abs=1e-4)
    assert isinstance(raw_result, MultiFORMResult)
    assert raw_result.getEventProbability() == probability


def test_type_errors(uncertain_space, f, snapshot):
    """Check the type errors when events are wrong."""
    system_form = OT_SystemFORM()
    problem = ReliabilityProblem(uncertain_space)
    with assert_exception(ValueError, snapshot):
        system_form.execute(problem)
