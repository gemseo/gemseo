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
"""Consistency between the different ways of assessing feasibility.

`Constraints.is_constraint_satisfied`
and `Constraints.is_point_feasible`
compare a value against its tolerance directly,
`value <= tolerance`
(or `abs(value) <= tolerance` for an equality constraint),
on an allocation-free hot path.

`Constraints.assess`
(hence `ConstraintAssessment.from_value`)
is a separate implementation
which uses the same comparison
and also reports the violation measure;
`Constraints.get_number_of_unsatisfied_constraints`
and `OptimizationHistory.check_design_point_is_feasible`
are built on it.

These tests check, on random points, that they agree.
"""

from __future__ import annotations

from enum import StrEnum
from enum import auto
from math import isinf
from typing import TYPE_CHECKING
from typing import TypeAlias

import pytest
from numpy import array
from numpy import inf
from numpy import nan
from numpy import nextafter
from numpy.random import default_rng

from gemseo.core.function.array_function import ArrayFunction
from gemseo.optimization.problem import OptimizationProblem
from gemseo.space.design import DesignSpace

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.random import Generator

    from gemseo.util.typing import RealArray

_EQ = ArrayFunction.ConstraintType.EQ
_INEQ = ArrayFunction.ConstraintType.INEQ
_INEQUALITY_TOLERANCE = 1e-3
# Distinct from the inequality tolerance,
# so that swapping the two tolerances in `Constraints.__get_tolerance`
# would be caught by these tests.
_EQUALITY_TOLERANCE = 2 * _INEQUALITY_TOLERANCE
_N_SEEDS = 10
_N_CASES_PER_SEED = 50
_X = array([0.5])
"""The design point at which the constraint values are stored."""

_Specs: TypeAlias = list[tuple[str, ArrayFunction.ConstraintType, int]]


class _Defect(StrEnum):
    """The defect of the value of one constraint of a generated point."""

    NAN = auto()
    """The value contains NaN."""

    MISSING = auto()
    """The value is missing."""


def _tolerance_of(constraint_type: ArrayFunction.ConstraintType) -> float:
    """Return the tolerance of a type of constraint.

    Args:
        constraint_type: The type of the constraint.

    Returns:
        The tolerance.
    """
    return _EQUALITY_TOLERANCE if constraint_type == _EQ else _INEQUALITY_TOLERANCE


def _make_problem(specs: _Specs, point: Mapping[str, RealArray]) -> OptimizationProblem:
    """Create an optimization problem and store a point in its database.

    Args:
        specs: The name, type and dimension of each constraint.
        point: The map from the names of the present constraints to their values.

    Returns:
        The optimization problem.
    """
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=0.0, upper_bound=1.0, value=0.5)
    problem = OptimizationProblem(design_space)
    problem.tolerances.inequality = _INEQUALITY_TOLERANCE
    problem.tolerances.equality = _EQUALITY_TOLERANCE
    problem.objective = ArrayFunction(lambda x: x, name="f")
    for name, constraint_type, _ in specs:
        problem.add_constraint(
            ArrayFunction(lambda x: x, name=name), constraint_type=constraint_type
        )

    problem.database.store(_X, {"f": array([0.5]), **point})
    return problem


def _generate_values(
    rng: Generator, constraint_type: ArrayFunction.ConstraintType, dimension: int
) -> RealArray:
    """Generate random values of a constraint around its tolerance.

    Args:
        rng: The random number generator.
        constraint_type: The type of the constraint.
        dimension: The dimension of the constraint.

    Returns:
        The values of the constraint.
    """
    tolerance = _tolerance_of(constraint_type)
    if rng.random() < 0.15:
        # Exactly at the tolerance, or at the closest representable
        # values around it, including the signed infinities.
        boundary_values = (
            tolerance,
            -tolerance,
            nextafter(tolerance, inf),
            -nextafter(tolerance, inf),
            inf,
            -inf,
        )
        return array([
            boundary_values[rng.integers(0, len(boundary_values))]
            for _ in range(dimension)
        ])

    return rng.uniform(-3 * tolerance, 3 * tolerance, size=dimension)


def _generate_case(
    rng: Generator, defect: _Defect | None = None
) -> tuple[_Specs, dict[str, RealArray]]:
    """Generate random constraints and a random point around their tolerances.

    Args:
        rng: The random number generator.
        defect: The defect of the value of one constraint, chosen at random.
            If `None`, no value has a defect.

    Returns:
        The name, type and dimension of each constraint,
        and the map from the names of the present constraints to their values.
    """
    specs = [
        (f"c{i}", _EQ if rng.random() < 0.5 else _INEQ, int(rng.integers(1, 4)))
        for i in range(int(rng.integers(1, 4)))  # 1 to 3 constraints
    ]
    defective_index = int(rng.integers(0, len(specs)))
    point = {}
    for index, (name, constraint_type, dimension) in enumerate(specs):
        if defect is _Defect.MISSING and index == defective_index:
            continue

        values = _generate_values(rng, constraint_type, dimension)
        if defect is _Defect.NAN and index == defective_index:
            values[int(rng.integers(0, dimension))] = nan

        point[name] = values

    return specs, point


@pytest.mark.parametrize("seed", range(_N_SEEDS))
def test_consistency_without_nan_nor_missing_value(seed) -> None:
    """Check that the feasibility functions agree without NaN nor missing value."""
    rng = default_rng(seed)
    for _ in range(_N_CASES_PER_SEED):
        specs, point = _generate_case(rng)
        problem = _make_problem(specs, point)
        stored = problem.database.get(_X)
        constraints = problem.constraints
        case = f"specs={specs}, point={point}"

        is_feasible = constraints.is_point_feasible(stored)
        check_feasible, check_violation = (
            problem.history.check_design_point_is_feasible(_X)
        )
        assessment = constraints.assess(stored)
        n_unsatisfied = constraints.get_number_of_unsatisfied_constraints(point)

        assert check_feasible == is_feasible, (
            f"check_design_point_is_feasible and is_point_feasible disagree: {case}"
        )
        assert assessment.are_satisfied == is_feasible, (
            f"assess and is_point_feasible disagree: {case}"
        )
        assert (check_violation == 0.0) == is_feasible, (
            "check_design_point_is_feasible (violation) and is_point_feasible "
            f"disagree: {case}"
        )
        assert (assessment.squared_violation == 0.0) == is_feasible, (
            f"assess (squared violation) and is_point_feasible disagree: {case}"
        )
        assert (assessment.max_violation == 0.0) == is_feasible, (
            f"assess (max violation) and is_point_feasible disagree: {case}"
        )
        assert check_violation == assessment.squared_violation, (
            f"check_design_point_is_feasible and assess disagree: {case}"
        )
        assert (n_unsatisfied == 0) == is_feasible, (
            f"get_number_of_unsatisfied_constraints and is_point_feasible "
            f"disagree: {case}"
        )
        assert n_unsatisfied == assessment.n_violated_components, (
            f"get_number_of_unsatisfied_constraints and assess disagree: {case}"
        )


@pytest.mark.parametrize("seed", range(_N_SEEDS))
def test_consistency_with_nan_value(seed) -> None:
    """Check that the feasibility functions agree with a NaN value."""
    rng = default_rng(seed)
    for _ in range(_N_CASES_PER_SEED):
        specs, point = _generate_case(rng, _Defect.NAN)
        problem = _make_problem(specs, point)
        stored = problem.database.get(_X)
        constraints = problem.constraints
        case = f"specs={specs}, point={point}"

        is_feasible = constraints.is_point_feasible(stored)
        check_feasible, check_violation = (
            problem.history.check_design_point_is_feasible(_X)
        )
        assessment = constraints.assess(stored)
        n_unsatisfied = constraints.get_number_of_unsatisfied_constraints(point)

        assert not is_feasible, f"is_point_feasible is True: {case}"
        assert check_feasible == is_feasible, (
            f"check_design_point_is_feasible and is_point_feasible disagree: {case}"
        )
        assert assessment.are_satisfied == is_feasible, (
            f"assess and is_point_feasible disagree: {case}"
        )
        assert check_violation == inf, (
            f"check_design_point_is_feasible gives a finite violation: {case}"
        )
        assert check_violation == assessment.squared_violation, (
            f"check_design_point_is_feasible and assess disagree: {case}"
        )
        assert isinf(assessment.max_violation), (
            f"assess gives a finite max violation: {case}"
        )
        assert n_unsatisfied >= 1, (
            f"get_number_of_unsatisfied_constraints ignores the NaN: {case}"
        )
        assert n_unsatisfied == assessment.n_violated_components, (
            f"get_number_of_unsatisfied_constraints and assess disagree: {case}"
        )


@pytest.mark.parametrize("seed", range(_N_SEEDS))
def test_consistency_with_missing_value(seed) -> None:
    """Check that the feasibility functions agree with a missing value."""
    rng = default_rng(seed)
    for _ in range(_N_CASES_PER_SEED):
        specs, point = _generate_case(rng, _Defect.MISSING)
        problem = _make_problem(specs, point)
        stored = problem.database.get(_X)
        constraints = problem.constraints
        case = f"specs={specs}, point={point}"

        is_feasible = constraints.is_point_feasible(stored)
        check_feasible, check_violation = (
            problem.history.check_design_point_is_feasible(_X)
        )
        assessment = constraints.assess(stored)
        n_unsatisfied = constraints.get_number_of_unsatisfied_constraints(point)

        assert not is_feasible, f"is_point_feasible is True: {case}"
        assert check_feasible == is_feasible, (
            f"check_design_point_is_feasible and is_point_feasible disagree: {case}"
        )
        assert assessment.are_satisfied == is_feasible, (
            f"assess and is_point_feasible disagree: {case}"
        )
        assert check_violation == inf, (
            f"check_design_point_is_feasible gives a finite violation: {case}"
        )
        assert check_violation == assessment.squared_violation, (
            f"check_design_point_is_feasible and assess disagree: {case}"
        )
        missing_names = tuple(name for name, _, _ in specs if name not in point)
        assert assessment.missing_names == missing_names, (
            f"assess gives wrong missing names: {case}"
        )
        assert n_unsatisfied == assessment.n_violated_components, (
            f"get_number_of_unsatisfied_constraints and assess disagree: {case}"
        )


@pytest.mark.parametrize(
    ("constraint_type", "value", "is_feasible"),
    [
        (_EQ, 0.5 * _EQUALITY_TOLERANCE + 10j, True),
        (_EQ, 2 * _EQUALITY_TOLERANCE + 0j, False),
        (_INEQ, _INEQUALITY_TOLERANCE + 10j, True),
        (_INEQ, 2 * _INEQUALITY_TOLERANCE - 10j, False),
    ],
    ids=[
        "equality-small-real-part-large-imaginary-part",
        "equality-large-real-part",
        "inequality-real-part-at-tolerance-large-imaginary-part",
        "inequality-large-real-part-negative-imaginary-part",
    ],
)
def test_consistency_with_complex_value(constraint_type, value, is_feasible) -> None:
    """Check that the feasibility functions assess a complex value on its real part."""
    point = {"c": array([value])}
    problem = _make_problem([("c", constraint_type, 1)], point)
    stored = problem.database.get(_X)
    constraints = problem.constraints

    assert constraints.is_point_feasible(stored) == is_feasible
    assert constraints.assess(stored).are_satisfied == is_feasible
    assert (constraints.get_number_of_unsatisfied_constraints(point) == 0) == (
        is_feasible
    )
    assert problem.history.check_design_point_is_feasible(_X)[0] == is_feasible
