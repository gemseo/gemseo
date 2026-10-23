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

from operator import ge
from operator import gt
from operator import le
from operator import lt
from typing import TYPE_CHECKING
from typing import Final
from typing import TypeAlias

import pytest
from numpy import sqrt
from openturns import MultiFORMResult
from scipy.stats import multivariate_normal
from scipy.stats import norm

from gemseo.core.function.array_function import ArrayFunction
from gemseo.space.random import RandomSpace
from gemseo.uncertainty.distribution.openturns.normal_settings import (
    OTNormalDistribution_Settings,
)
from gemseo.uncertainty.distribution.openturns.uniform_settings import (
    OTUniformDistribution_Settings,
)
from gemseo.uncertainty.reliability.openturns.form import OT_FORM
from gemseo.uncertainty.reliability.openturns.optimizer import BaseOTOptimizer
from gemseo.uncertainty.reliability.openturns.optimizer import OTNLopt
from gemseo.uncertainty.reliability.openturns.system_form import OT_SystemFORM
from gemseo.uncertainty.reliability.openturns.system_form import (
    _collinear_error_fragment,
)
from gemseo.uncertainty.reliability.openturns.system_form_settings import (
    OT_SystemFORM_Settings,
)
from gemseo.uncertainty.reliability.problem import ReliabilityProblem
from gemseo.util.testing.helper import assert_exception

if TYPE_CHECKING:
    from collections.abc import Callable

    from gemseo.uncertainty.reliability.event import Event
    from gemseo.uncertainty.reliability.event_variable import EventVariable
    from gemseo.uncertainty.reliability.openturns.multi_form_result import (
        MultiFORMResult as GemseoMultiFORMResult,
    )
    from gemseo.uncertainty.reliability.result import ReliabilityResult
    from gemseo.util.typing import RealArray

_Comparison: TypeAlias = "Callable[[EventVariable, float], Event]"
"""A comparison of an event variable to a threshold, e.g. `operator.gt`."""


def f_1(u: RealArray) -> RealArray:
    """The function returning the first input.

    Args:
        u: The input value.

    Returns:
        The output value.
    """
    return u[[0]]


def f_2(u: RealArray) -> RealArray:
    """The function returning the second input.

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
def random_space() -> RandomSpace:
    """The uncertainty space defined by the random vector u=(u1,u2), ui ~ U([0,1])."""
    parameter_space = RandomSpace()
    parameter_space.add_variable(
        "u", OTUniformDistribution_Settings(), OTUniformDistribution_Settings()
    )
    return parameter_space


@pytest.fixture(
    scope="module",
    params=[
        None,
        *(
            OT_SystemFORM_Settings(optimizer=cls())
            for cls in BaseOTOptimizer.__subclasses__()
            if cls != OTNLopt
            # RuntimeError: Exception :
            # Obtained design point is not on the limit state:
            # its image by the limit state function is 0.749731,
            # which is incompatible with the threshold: 0.75
            # considering the limit state tolerance of the optimization algorithm: 1e-05
        ),
    ],
    ids=lambda settings: (
        "default" if settings is None else settings.optimizer.__class__.__name__
    ),
)
def settings(request) -> OT_SystemFORM_Settings | None:
    """The settings of SystemFORM, if any."""
    return request.param


_comparisons_to_probability: Final[dict[tuple[_Comparison, _Comparison], float]] = {
    # P[y>0.75]P[z>0.75]=0.25²
    (gt, gt): 0.0625,
    # P[y<0.75]P[z>0.75]=0.75*0.25
    (lt, gt): 0.1875,
    # P[y>0.75]P[z<0.75]=0.25*0.75
    (gt, lt): 0.1875,
    # P[y<0.75]P[z<0.75]=0.75²
    (lt, lt): 0.5625,
}
"""The map from the comparisons of y and z to 0.75 to their joint probability."""


@pytest.fixture(
    scope="module",
    params=list(_comparisons_to_probability),
    ids=["y_above-z_above", "y_below-z_above", "y_above-z_below", "y_below-z_below"],
)
def comparisons(request) -> tuple[_Comparison, _Comparison]:
    """The comparisons of y and z to 0.75."""
    return request.param


@pytest.fixture(scope="module")
def intersection_results(
    function_1, function_2, random_space, settings, comparisons
) -> dict[str, ReliabilityResult]:
    """The results of the SystemFORM analysis of the intersection of two events.

    The event "a" is the intersection of the comparisons of y and z to 0.75.
    """
    problem = ReliabilityProblem(random_space)
    y, z = problem.get_event_variables(function_1, function_2)
    compare_y, compare_z = comparisons
    problem.add_event(compare_y(y, 0.75) & compare_z(z, 0.75), event_name="a")
    return OT_SystemFORM().execute(problem, settings=settings)


def test_execute_returns_one_result_named_after_the_event(intersection_results):
    """Check that the result of the event is the only one and is named after it."""
    assert len(intersection_results) == 1
    assert intersection_results["a"].name == "a"


def test_execute_probability_of_intersection(intersection_results, comparisons):
    """Check the probability of an intersection of two events."""
    probability = intersection_results["a"].probability
    expected = _comparisons_to_probability[comparisons]
    assert probability == pytest.approx(expected, abs=1e-4)


def test_execute_raw_result_is_multiform_result(intersection_results):
    """Check that the raw result is an OpenTURNS MultiFORMResult."""
    assert isinstance(intersection_results["a"].raw_result, MultiFORMResult)


def test_execute_result_matches_raw_result(intersection_results):
    """Check that the probability and reliability index come from the raw result."""
    result = intersection_results["a"]
    raw_result = result.raw_result
    assert raw_result.getEventProbability() == result.probability
    assert raw_result.getGeneralisedReliabilityIndex() == result.reliability_index


def test_execute_without_event(random_space, snapshot):
    """Check that a problem without event raises a ValueError."""
    problem = ReliabilityProblem(random_space)
    with assert_exception(ValueError, snapshot):
        OT_SystemFORM().execute(problem)


def f_h(u: RealArray) -> RealArray:
    """The function h(u)=u1+u2.

    Args:
        u: The input value.

    Returns:
        The output value.
    """
    return u[[0]] + u[[1]]


def f_g1(u: RealArray) -> RealArray:
    """The function g1(u)=u1-u2+u3.

    Args:
        u: The input value.

    Returns:
        The output value.
    """
    return u[[0]] - u[[1]] + u[[2]]


def f_g2(u: RealArray) -> RealArray:
    """The function g2(u)=u2+u3.

    Args:
        u: The input value.

    Returns:
        The output value.
    """
    return u[[1]] + u[[2]]


@pytest.fixture(scope="module")
def normal_random_space() -> RandomSpace:
    """The uncertainty space of u=(u1,u2,u3), ui ~ N(0,1) independent."""
    parameter_space = RandomSpace()
    parameter_space.add_variable(
        "u",
        OTNormalDistribution_Settings(),
        OTNormalDistribution_Settings(),
        OTNormalDistribution_Settings(),
    )
    return parameter_space


def _create_problem(
    random_space: RandomSpace,
    compare_first_h: _Comparison,
    compare_second_h: _Comparison,
) -> ReliabilityProblem:
    """Create a problem whose event compares h to 0.5 twice.

    The event is (h compared to 0.5 and g1>0.4) or (h compared to 0.5 and g2>0.3),
    so the elementary event on h appears once in each branch of the union,
    with its own comparison.

    Args:
        random_space: The uncertainty space.
        compare_first_h: The comparison of h to 0.5 in the first branch,
            e.g. `operator.gt` for h>0.5.
        compare_second_h: The comparison of h to 0.5 in the second branch,
            e.g. `operator.ge` for h>=0.5.

    Returns:
        The problem.
    """
    problem = ReliabilityProblem(random_space)
    h, g1, g2 = problem.get_event_variables(
        ArrayFunction(f_h, name="h"),
        ArrayFunction(f_g1, name="g1"),
        ArrayFunction(f_g2, name="g2"),
    )
    first_branch = compare_first_h(h, 0.5) & (g1 > 0.4)
    second_branch = compare_second_h(h, 0.5) & (g2 > 0.3)
    problem.add_event(first_branch | second_branch, event_name="a")
    return problem


@pytest.fixture(
    scope="module",
    params=[(gt, gt), (gt, ge), (lt, lt), (lt, le)],
    ids=[
        "above-same",
        "above-strict_and_non_strict",
        "below-same",
        "below-strict_and_non_strict",
    ],
)
def repeated_h_comparisons(request) -> tuple[_Comparison, _Comparison]:
    """The comparisons of the two occurrences of h to 0.5, on the same side."""
    return request.param


@pytest.fixture(scope="module")
def repeated_event_result(
    normal_random_space, repeated_h_comparisons
) -> GemseoMultiFORMResult:
    """The result of the SystemFORM analysis of the problem with a repeated event."""
    compare_first_h, compare_second_h = repeated_h_comparisons
    problem = _create_problem(
        normal_random_space,
        compare_first_h=compare_first_h,
        compare_second_h=compare_second_h,
    )
    return OT_SystemFORM().execute(problem)["a"]


def test_execute_analyzes_repeated_event_once(repeated_event_result):
    """Check that a repeated elementary event is analyzed once."""
    # The event is written with four elementary events,
    # the one on h appearing twice on the same side of 0.5,
    # so SystemFORM must run three FORM analyses,
    # one per distinct limit state and side,
    # whether the comparisons are strict or not.
    assert len(repeated_event_result.form_results) == 3


def test_execute_probability_with_repeated_event(
    repeated_h_comparisons, repeated_event_result
):
    """Check the probability of an event whose elementary event is repeated."""
    # The limit states are linear in standard normal variables,
    # so FORM, and so SystemFORM, is exact up to the numerical integration,
    # and it does not distinguish a strict comparison from a non-strict one.
    # (h,g1,g2) is a centred Gaussian vector
    # with variances 2, 3 and 2,
    # cov(h,g2)=1 and cov(h,g1)=cov(g1,g2)=0,
    # so g1 is independent of (h,g2).
    h_g2 = multivariate_normal(cov=[[2.0, 1.0], [1.0, 2.0]])
    p_g1_below = norm.cdf(0.4 / sqrt(3))
    # Above, the event is h>0.5 and (g1>0.4 or g2>0.3),
    # so its probability is P(h>0.5) minus P(h>0.5, g1<0.4, g2<0.3),
    # which is P(g1<0.4) times P(h>0.5, g2<0.3),
    # i.e. times P(g2<0.3) minus P(h<0.5, g2<0.3).
    p_above = norm.sf(0.5 / sqrt(2)) - p_g1_below * (
        norm.cdf(0.3 / sqrt(2)) - h_g2.cdf([0.5, 0.3])
    )
    # Below, the event is h<0.5 and (g1>0.4 or g2>0.3),
    # so its probability is P(h<0.5) minus P(h<0.5, g1<0.4, g2<0.3),
    # which is P(g1<0.4) times P(h<0.5, g2<0.3).
    p_below = norm.cdf(0.5 / sqrt(2)) - p_g1_below * h_g2.cdf([0.5, 0.3])
    side_to_probability = {gt: p_above, lt: p_below}
    expected = side_to_probability[repeated_h_comparisons[0]]
    assert repeated_event_result.probability == pytest.approx(expected, abs=1e-4)


@pytest.fixture(scope="module")
def opposite_sides_result(normal_random_space) -> GemseoMultiFORMResult:
    """The result of the SystemFORM analysis of the problem with opposite sides.

    The event is (h>0.5 and g1>0.4) or (h<0.5 and g2>0.3).
    """
    problem = _create_problem(
        normal_random_space, compare_first_h=gt, compare_second_h=lt
    )
    return OT_SystemFORM().execute(problem)["a"]


def test_execute_does_not_share_opposite_sides(opposite_sides_result):
    """Check that the two sides of a limit state are analyzed separately."""
    # The event is written with four elementary events,
    # h>0.5 and h<0.5 sharing their limit state but not their side,
    # so SystemFORM must run four FORM analyses.
    assert len(opposite_sides_result.form_results) == 4


def test_execute_probability_with_opposite_sides(opposite_sides_result):
    """Check the probability of an event using both sides of a limit state."""
    # The limit states are linear in standard normal variables,
    # so FORM, and so SystemFORM, is exact up to the numerical integration.
    # The branches h>0.5 and g1>0.4, and h<0.5 and g2>0.3, are disjoint,
    # so the probability is the sum of their probabilities.
    # (h,g1,g2) is a centred Gaussian vector
    # with variances 2, 3 and 2,
    # cov(h,g2)=1 and cov(h,g1)=cov(g1,g2)=0,
    # so h and g1 are independent
    # and P(h<0.5 and g2>0.3) is P(h<0.5) minus P(h<0.5 and g2<0.3).
    h_g2 = multivariate_normal(cov=[[2.0, 1.0], [1.0, 2.0]])
    exact = (
        norm.sf(0.5 / sqrt(2)) * norm.sf(0.4 / sqrt(3))
        + norm.cdf(0.5 / sqrt(2))
        - h_g2.cdf([0.5, 0.3])
    )
    assert opposite_sides_result.probability == pytest.approx(exact, abs=1e-4)


_singular_correlation_message: Final[str] = (
    f"InvalidArgumentException : Error: {_collinear_error_fragment} "
    "must be in less or equal to 1, here delta=1"
)
"""The message of the OpenTURNS error raised on a singular correlation matrix.

Its wording is pinned on purpose,
because OT_SystemFORM translates the errors containing this fragment.
"""


@pytest.mark.parametrize(
    ("method_name", "message", "exception"),
    [
        ("_run_ot_algo", _singular_correlation_message, ValueError),
        ("_run_ot_algo", "Another error.", TypeError),
        # This case guards that only the errors raised by the OpenTURNS run
        # are translated,
        # and not the ones raised elsewhere with the same fragment.
        ("_create_ot_event", f"stub: {_collinear_error_fragment}", TypeError),
    ],
    ids=["singular_correlation", "other_error", "outside_run"],
)
def test_execute_translates_singular_correlation_error(
    normal_random_space, monkeypatch, snapshot, method_name, message, exception
):
    """Check that only the singular correlation error of the run is translated."""

    def raise_type_error(*args, **kwargs):
        raise TypeError(message)

    monkeypatch.setattr(OT_FORM, method_name, raise_type_error)
    problem = _create_problem(
        normal_random_space, compare_first_h=gt, compare_second_h=gt
    )
    with assert_exception(exception, snapshot):
        OT_SystemFORM().execute(problem)
