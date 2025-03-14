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
from logging import WARNING

from gemseo import configure_logger
from gemseo import create_discipline
from gemseo import create_scenario
from gemseo import execute_post
from gemseo.algos.opt.nlopt.settings.nlopt_cobyla_settings import NLOPT_COBYLA_Settings
from gemseo.algos.opt.scipy_local.settings.slsqp import SLSQP_Settings
from gemseo.problems.mdo.sobieski.core.design_space import SobieskiDesignSpace

configure_logger()

# %%
# Instantiate the  disciplines
# ----------------------------
# First, we instantiate the four disciplines of the use case:
# :class:`.SobieskiPropulsion`,
# :class:`.SobieskiAerodynamics`,
# :class:`.SobieskiMission`
# and :class:`.SobieskiStructure`.
propu, aero, mission, struct = create_discipline([
    "SobieskiPropulsion",
    "SobieskiAerodynamics",
    "SobieskiMission",
    "SobieskiStructure",
])

# %%
# Build, execute and post-process the scenario
# --------------------------------------------
# Then, we build the scenario which links the disciplines
# with the formulation and the optimization algorithm. Here, we use the
# :class:`.BiLevel` formulation. We tell the scenario to minimize -y_4
# instead of minimizing y_4 (range), which is the default option.
#
# We need to define the design space.
design_space = SobieskiDesignSpace()

# %%
# Then, we build a sub-scenario for each strongly coupled disciplines,
# using the following algorithm, maximum number of iterations and
# algorithm settings:

slsqp_settings = SLSQP_Settings(
    max_iter=30,
    xtol_rel=1e-7,
    xtol_abs=1e-7,
    ftol_rel=1e-7,
    ftol_abs=1e-7,
    ineq_tolerance=1e-4,
)

cobyla_settings = NLOPT_COBYLA_Settings(
    max_iter=50,
    xtol_rel=1e-7,
    xtol_abs=1e-7,
    ftol_rel=1e-7,
    ftol_abs=1e-7,
    ineq_tolerance=1e-4,
)

# %%
# Build a sub-scenario for Propulsion
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# This sub-scenario will minimize SFC.
sc_prop = create_scenario(
    propu,
    "y_34",
    design_space.filter("x_3", copy=True),
    name="PropulsionScenario",
    formulation_name="DisciplinaryOpt",
)
sc_prop.set_algorithm(slsqp_settings)
sc_prop.add_constraint("g_3", constraint_type="ineq")

# %%
# Build a sub-scenario for Aerodynamics
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# This sub-scenario will minimize L/D.
sc_aero = create_scenario(
    aero,
    "y_24",
    design_space.filter("x_2", copy=True),
    name="AerodynamicsScenario",
    maximize_objective=True,
    formulation_name="DisciplinaryOpt",
)
sc_aero.set_algorithm(slsqp_settings)
sc_aero.add_constraint("g_2", constraint_type="ineq")

# %%
# Build a sub-scenario for Structure
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# This sub-scenario will maximize
# log(aircraft total weight / (aircraft total weight - fuel weight)).
sc_str = create_scenario(
    struct,
    "y_11",
    design_space.filter("x_1", copy=True),
    name="StructureScenario",
    maximize_objective=True,
    formulation_name="DisciplinaryOpt",
)
sc_str.add_constraint("g_1", constraint_type="ineq")
sc_str.set_algorithm(slsqp_settings)

# %%
# Build a scenario for Mission
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# This scenario is based on the three previous sub-scenarios and on the
# Mission and aims to maximize the range (Breguet).
system_scenario = create_scenario(
    [sc_prop, sc_aero, sc_str, mission],
    "y_4",
    design_space.filter("x_shared", copy=True),
    apply_cstr_tosub_scenarios=False,
    parallel_scenarios=False,
    multithread_scenarios=True,
    main_mda_settings={"tolerance": 1e-14, "max_mda_iter": 30},
    maximize_objective=True,
    sub_scenarios_log_level=WARNING,
    formulation_name="BiLevel",
)
system_scenario.add_constraint(["g_1", "g_2", "g_3"], constraint_type="ineq")

# %%
# .. tip::
#
#    When running BiLevel scenarios, it is interesting to access the optimization
#    history of the sub-scenarios for each system iteration. By default, the setting
#    ``keep_opt_history`` is set to ``True``. This allows you to store in memory the
#    databases of the sub-scenarios (see the last section of this example for more
#    details).
#    In some cases, storing the databases in memory can take up too much space and cause
#    performance issues. In these cases, set ``keep_opt_history=False`` and save the
#    databases to the disk using ``save_opt_history=True``.

# %%
# Visualize the XDSM
# ^^^^^^^^^^^^^^^^^^
# Generate the XDSM on the fly:
#
# - ``log_workflow_status=True`` will log the status of the workflow  in the console,
# - ``save_html`` (default ``True``) will generate a self-contained HTML file,
#   that can be automatically opened using ``show_html=True``.
system_scenario.xdsmize(save_html=False, pdf_build=False)

# %%
# Execute the main scenario
# ^^^^^^^^^^^^^^^^^^^^^^^^^
system_scenario.execute(cobyla_settings)

# %%
# Plot the history of the MDA residuals
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# For the first MDA:
system_scenario.formulation.mda1.plot_residual_history(save=False, show=True)

# %%
# For the second MDA:
system_scenario.formulation.mda2.plot_residual_history(save=False, show=True)

# %%
# Plot the system optimization history view
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
system_scenario.post_process(post_name="OptHistoryView", save=False, show=True)

# %%
# Plot the structure optimization histories of the 2 first iterations
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
struct_databases = system_scenario.formulation.scenario_adapters[2].databases
for database in struct_databases[:2]:
    opt_problem = deepcopy(sc_str.formulation.optimization_problem)
    opt_problem.database = database
    execute_post(opt_problem, post_name="OptHistoryView", save=False, show=True)

for disc in [propu, aero, mission, struct]:
    print(f"{disc.name}: {disc.execution_statistics.n_executions} calls.")
