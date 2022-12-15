# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This work is licensed under a BSD 0-Clause License.
#
# Permission to use, copy, modify, and/or distribute this software
# for any purpose with or without fee is hereby granted.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL
# WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL
# THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT,
# OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING
# FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
# NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION
# WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
# Contributors:
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: François Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
BiLevel-based MDO on the Sobieski SSBJ test case
================================================
"""
from __future__ import annotations

from copy import deepcopy

from gemseo.api import configure_logger
from gemseo.api import create_discipline
from gemseo.api import create_scenario
from gemseo.api import execute_post
from gemseo.problems.sobieski.core.problem import SobieskiProblem

configure_logger()

##############################################################################
# Instantiate the  disciplines
# ----------------------------
# First, we instantiate the four disciplines of the use case:
# :class:`~gemseo.problems.sobieski.disciplines.SobieskiPropulsion`,
# :class:`~gemseo.problems.sobieski.disciplines.SobieskiAerodynamics`,
# :class:`~gemseo.problems.sobieski.disciplines.SobieskiMission`
# and :class:`~gemseo.problems.sobieski.disciplines.SobieskiStructure`.
propu, aero, mission, struct = create_discipline(
    [
        "SobieskiPropulsion",
        "SobieskiAerodynamics",
        "SobieskiMission",
        "SobieskiStructure",
    ]
)

##############################################################################
# Build, execute and post-process the scenario
# --------------------------------------------
# Then, we build the scenario which links the disciplines
# with the formulation and the optimization algorithm. Here, we use the
# :class:`.BiLevel` formulation. We tell the scenario to minimize -y_4
# instead of minimizing y_4 (range), which is the default option.
#
# We need to define the design space.
design_space = SobieskiProblem().design_space

##############################################################################
# Then, we build a sub-scenario for each strongly coupled disciplines,
# using the following algorithm, maximum number of iterations and
# algorithm options:
algo_options = {
    "xtol_rel": 1e-7,
    "xtol_abs": 1e-7,
    "ftol_rel": 1e-7,
    "ftol_abs": 1e-7,
    "ineq_tolerance": 1e-4,
}
sub_sc_opts = {"max_iter": 30, "algo": "SLSQP", "algo_options": algo_options}
##############################################################################
# Build a sub-scenario for Propulsion
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# This sub-scenario will minimize SFC.
sc_prop = create_scenario(
    propu,
    "DisciplinaryOpt",
    "y_34",
    design_space=deepcopy(design_space).filter("x_3"),
    name="PropulsionScenario",
)
sc_prop.default_inputs = sub_sc_opts
sc_prop.add_constraint("g_3", constraint_type="ineq")

##############################################################################
# Build a sub-scenario for Aerodynamics
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# This sub-scenario will minimize L/D.
sc_aero = create_scenario(
    aero,
    "DisciplinaryOpt",
    "y_24",
    deepcopy(design_space).filter("x_2"),
    name="AerodynamicsScenario",
    maximize_objective=True,
)
sc_aero.default_inputs = sub_sc_opts
sc_aero.add_constraint("g_2", constraint_type="ineq")

##############################################################################
# Build a sub-scenario for Structure
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# This sub-scenario will maximize
# log(aircraft total weight / (aircraft total weight - fuel weight)).
sc_str = create_scenario(
    struct,
    "DisciplinaryOpt",
    "y_11",
    deepcopy(design_space).filter("x_1"),
    name="StructureScenario",
    maximize_objective=True,
)
sc_str.add_constraint("g_1", constraint_type="ineq")
sc_str.default_inputs = sub_sc_opts

##############################################################################
# Build a scenario for Mission
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# This scenario is based on the three previous sub-scenarios and on the
# Mission and aims to maximize the range (Breguet).
sub_disciplines = [sc_prop, sc_aero, sc_str] + [mission]
design_space = deepcopy(design_space).filter("x_shared")
system_scenario = create_scenario(
    sub_disciplines,
    "BiLevel",
    "y_4",
    design_space,
    apply_cstr_tosub_scenarios=False,
    parallel_scenarios=False,
    multithread_scenarios=True,
    tolerance=1e-14,
    max_mda_iter=30,
    maximize_objective=True,
)
system_scenario.add_constraint(["g_1", "g_2", "g_3"], "ineq")

# system_scenario.xdsmize(open_browser=True)
system_scenario.execute(
    {"max_iter": 50, "algo": "NLOPT_COBYLA", "algo_options": algo_options}
)

##############################################################################
# Plot the history of the MDA residuals
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# For the first MDA:
system_scenario.formulation.mda1.plot_residual_history(save=False, show=True)
# For the second MDA:
system_scenario.formulation.mda2.plot_residual_history(save=False, show=True)

##############################################################################
# Plot the system optimization history view
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
system_scenario.post_process("OptHistoryView", save=False, show=True)

##############################################################################
# Plot the structure optimization histories of the 2 first iterations
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

struct_databases = system_scenario.formulation.scenario_adapters[2].databases
for database in struct_databases[:2]:
    opt_problem = deepcopy(sc_str.formulation.opt_problem)
    opt_problem.database = database
    execute_post(opt_problem, "OptHistoryView", save=False, show=True)

for disc in [propu, aero, mission, struct]:
    print(f"{disc.name}: {disc.n_calls} calls.")
