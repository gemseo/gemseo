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

"""Fixtures shared by the directory manager tests."""

from __future__ import annotations

from math import sqrt
from typing import TYPE_CHECKING
from typing import Final

import pytest

from gemseo.space.util import get_value_and_bounds

if TYPE_CHECKING:
    from gemseo.optimization import OptimizationProblem
    from gemseo.optimization.core.base_optimization_library import (
        BaseOptimizationLibrary,
    )

golden_ratio_conjugate: Final[float] = (sqrt(5) - 1) / 2
"""The fractional part of the golden ratio, used as a low-discrepancy step.

Multiplying the 1-based sweep-point index by this irrational number and
taking the value modulo 1 produces a non-monotonic, well-spread sequence in
`[0, 1)` (0.618, 0.236, 0.854, 0.472, 0.090, ...): unlike an evenly spaced
sweep, no sweep point is systematically the smallest or the largest objective
value, so the optimum iteration used by the `KEEP_SOLUTION_ONLY` and
`KEEP_BASELINE_AND_SOLUTION` policies is unlikely to land on the first or the
last directory created for a scenario. The very first directory is always
x0, evaluated by `_pre_run` before any sweep point; it is not itself a sweep
point.
"""


def _run_deterministic_sweep(
    self: BaseOptimizationLibrary, problem: OptimizationProblem
) -> tuple[str, int]:
    """Evaluate a deterministic sweep of points instead of running the algorithm.

    This replaces the real solver iteration by evaluating up to
    `self._settings.max_iter` points regularly spread (in a low-discrepancy,
    non-monotonic order) over the design space, irrespective of the optimizer
    or of the SciPy/NLopt version: the produced directory tree (one directory
    per database iteration) no longer depends on optimizer internals, so it is
    stable across platforms and dependency versions.

    `_pre_run` evaluates x0 first, as database iteration 1. The driver then
    stops this loop through `MaxIterReachedException` on the evaluation
    exceeding `max_iter`, so the database holds x0 and at most
    `max_iter - 1` sweep points.

    Args:
        self: The bound optimization library instance.
        problem: The optimization problem to evaluate.

    Returns:
        A dummy termination message and status, as the real `_run` methods
        return. Only reached if the driver does not stop the loop first; kept
        to honour the `_run` contract.
    """
    _, l_b, u_b = get_value_and_bounds(problem.input_space)
    max_iter = self._settings.max_iter
    require_gradient = self.ALGORITHM_INFOS[self._algo_name].require_gradient
    constraints = self._get_right_sign_constraints(problem)

    for i in range(max_iter):
        t = ((i + 1) * golden_ratio_conjugate) % 1
        x = l_b + t * (u_b - l_b)
        problem.objective.evaluate(x)
        for constraint in constraints:
            constraint.evaluate(x)
        if require_gradient:
            problem.objective.jac(x)
            for constraint in constraints:
                constraint.jac(x)

    return "Deterministic mock", 0


@pytest.fixture(autouse=True)
def deterministic_optimizers(monkeypatch):
    """Make SciPy and NLopt optimizers deterministic.

    The directory tree produced by an MDO scenario (one directory per
    database iteration, kept or pruned by the cleanup policy under test)
    must not depend on the internals of the optimizer: replace both the
    SciPy-based and the NLopt-based `_run` methods by the same deterministic
    sweep, so the tree is identical across SciPy/NLopt versions and platforms.
    """
    monkeypatch.setattr(
        "gemseo.optimization.scipy_local.scipy_local.ScipyOpt._run",
        _run_deterministic_sweep,
    )
    monkeypatch.setattr(
        "gemseo.optimization.nlopt.nlopt.Nlopt._run",
        _run_deterministic_sweep,
    )
