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
from typing import TYPE_CHECKING

import pytest

from gemseo.problems.mdo.sobieski.core.design_space import SobieskiDesignSpace
from gemseo.problems.mdo.sobieski.disciplines import SobieskiAerodynamics
from gemseo.problems.mdo.sobieski.disciplines import SobieskiMission
from gemseo.problems.mdo.sobieski.disciplines import SobieskiPropulsion
from gemseo.problems.mdo.sobieski.disciplines import SobieskiStructure
from gemseo.scenarios.mdo_scenario import MDOScenario

if TYPE_CHECKING:
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
    max_iter: int = 50,
    include_weak_coupling_targets: bool = True,
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
        max_iter: The maximum number of iterations of the optimization algorithm.
        include_weak_coupling_targets: Whether to control all coupling variables.
            Otherwise, IDF will control the strong coupling variables only.

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
    design_space = SobieskiDesignSpace()
    if dtype == "complex128":
        design_space.to_complex()
    if remove_coupl_from_ds:
        for var in design_space.variable_names:
            if var.startswith("y_"):
                design_space.remove_variable(var)

    scenario = MDOScenario(
        disciplines,
        "y_4",
        design_space,
        formulation_name="IDF",
        normalize_constraints=normalize_cstr,
        n_processes=n_processes,
        use_threading=use_threading,
        maximize_objective=True,
        start_at_equilibrium=True,
        mda_chain_settings_for_start_at_equilibrium={"tolerance": 1e-8},
        include_weak_coupling_targets=include_weak_coupling_targets,
    )
    if linearize:
        scenario.set_differentiation_method()
    else:
        scenario.set_differentiation_method("complex_step", 1e-30)
    # Set the design constraints
    for c_name in ["g_1", "g_2", "g_3"]:
        scenario.add_constraint(c_name, constraint_type="ineq")

    scenario.execute(
        algo_name=algo,
        max_iter=max_iter,
        eq_tolerance=eq_tolerance,
        ineq_tolerance=ineq_tolerance,
    )

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
