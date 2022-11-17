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
"""Test helpers."""
from __future__ import annotations

from functools import partial

import pytest
from gemseo.core.mdo_scenario import MDOScenario
from gemseo.problems.sobieski.core.problem import SobieskiProblem
from gemseo.problems.sobieski.disciplines import SobieskiAerodynamics
from gemseo.problems.sobieski.disciplines import SobieskiMission
from gemseo.problems.sobieski.disciplines import SobieskiPropulsion
from gemseo.problems.sobieski.disciplines import SobieskiStructure
from numpy import ndarray


def build_and_run_idf_scenario_with_constraints(
    algo: str,
    linearize: bool = False,
    dtype: str = "complex128",
    normalize_cstr: bool = True,
    eq_tolerance: float = 1e-4,
    ineq_tolerance: float = 1e-3,
    remove_coupl_from_ds: bool = False,
    n_processes: int = 1,
    use_threading: bool = True,
    max_iter=50,
) -> tuple[ndarray, bool]:
    """Build and execute an :class:`.MDOScenario` with an IDF formulation.

    Args:
        algo: The optimization algorithm.
        linearize: If True, use the Jacobian defined in the disciplines. Otherwise,
            differentiate using complex step.
        dtype: The data type of the design space variables.
        normalize_cstr: Whether to normalize the constraints.
        eq_tolerance: The tolerance for the equality constraints.
        ineq_tolerance: The tolerance for the inequality constraints.
        remove_coupl_from_ds: Whether to remove the coupling variables from the
            design space.
        n_processes: The number of processes in which to run the formulation.
        use_threading: Whether to use multi-threading or multi-processing when running
            the formulation in parallel (n_processes > 1).

    Returns:
        The objective value after the execution of the scenario and whether the
            optimization result is feasible.
    """
    disciplines = [
        SobieskiStructure(dtype),
        SobieskiPropulsion(dtype),
        SobieskiAerodynamics(dtype),
        SobieskiMission(dtype),
    ]
    design_space = SobieskiProblem().design_space
    if dtype == "complex128":
        design_space.to_complex()
    if remove_coupl_from_ds:
        for var in design_space.variables_names:
            if var.startswith("y_"):
                design_space.remove_variable(var)

    scenario = MDOScenario(
        disciplines,
        "IDF",
        "y_4",
        design_space=design_space,
        normalize_constraints=normalize_cstr,
        n_processes=n_processes,
        use_threading=use_threading,
        maximize_objective=True,
        start_at_equilibrium=True,
    )
    if linearize:
        scenario.set_differentiation_method()
    else:
        scenario.set_differentiation_method("complex_step", 1e-30)
    # Set the design constraints
    for c_name in ["g_1", "g_2", "g_3"]:
        scenario.add_constraint(c_name, "ineq")

    run_inputs = {
        "max_iter": max_iter,
        "algo": algo,
        "algo_options": {
            "eq_tolerance": eq_tolerance,
            "ineq_tolerance": ineq_tolerance,
        },
    }

    scenario.execute(run_inputs)

    obj_opt = scenario.optimization_result.f_opt
    is_feasible = scenario.optimization_result.is_feasible
    return -obj_opt, is_feasible


@pytest.fixture
def generate_idf_scenario():
    """Wrap an :class:`.MDOScenario` with an IDF formulation.

    Returns:
        A wrapped :class:`.MDOScenario` for which a series of options can be
            passed to customize it.
    """
    return partial(build_and_run_idf_scenario_with_constraints)
