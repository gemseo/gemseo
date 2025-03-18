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
#        :author: Fabian Castañeda
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
BiLevel BCD-based MDO on the Sobieski SSBJ test case
================================================
"""

# %%
# .. note::
#
#    There are several variants of the BiLevel BCD formulation; this example shows
#    the implementation of the BiLevel BCD MDF (BL-BCD-MDF) for other variants please
#    refer to :cite:`david:hal-04758286`.

from __future__ import annotations

from copy import deepcopy

from gemseo import configure_logger
from gemseo import execute_post
from gemseo.algos.opt.nlopt.settings.nlopt_cobyla_settings import NLOPT_COBYLA_Settings
from gemseo.algos.opt.scipy_local.settings.slsqp import SLSQP_Settings
from gemseo.formulations.bilevel_bcd_settings import BiLevel_BCD_Settings
from gemseo.formulations.mdf_settings import MDF_Settings
from gemseo.mda.gauss_seidel_settings import MDAGaussSeidel_Settings
from gemseo.problems.mdo.sobieski.core.design_space import SobieskiDesignSpace
from gemseo.problems.mdo.sobieski.disciplines import SobieskiAerodynamics
from gemseo.problems.mdo.sobieski.disciplines import SobieskiMission
from gemseo.problems.mdo.sobieski.disciplines import SobieskiPropulsion
from gemseo.problems.mdo.sobieski.disciplines import SobieskiStructure
from gemseo.scenarios.mdo_scenario import MDOScenario

configure_logger()

# %%
# Instantiate the  disciplines
# ----------------------------
# First, we instantiate the four disciplines of the use case:
# :class:`.SobieskiPropulsion`,
# :class:`.SobieskiAerodynamics`,
# :class:`.SobieskiMission`
# and :class:`.SobieskiStructure`.
propulsion_disc = SobieskiPropulsion()
aerodynamics_disc = SobieskiAerodynamics()
structure_disc = SobieskiStructure()
mission_disc = SobieskiMission()

# %%
# Since they are going to be our disciplines for the sub-scenarios, we'll call them
# sub-disciplines.

sub_disciplines = [structure_disc, propulsion_disc, aerodynamics_disc, mission_disc]

# %%
# Build the scenario
# ----------------------
# We build the scenario that allows to create the optimization problem from
# the disciplines and the formulation.
# Here, we use the :class:`.BiLevelBCD` formulation.
# We need to define the design space.

design_space = SobieskiDesignSpace()
# %%
# For this formulation, we need to define the optimization sub-scenarios from
# all sub-disciplines coupled together. Each sub-scenario optimizes its own design
# variable according to the corresponding constraint and the objective y_4 (range)
# which we are maximizing.

# %%
# Define Sub-scenario settings model
# --------------------------------------
# The setting for all sub-scenarios is the same, so we can define a global, settings
# model to be used by each sub-scenario.

sub_scenario_settings = MDF_Settings(
    main_mda_name="MDAGaussSeidel",
)
sc_algo_settings = SLSQP_Settings(max_iter=50)

# %%
# Build the Propulsion Sub-scenario
# -----------------------------------
# This sub-scenario will optimize the propulsion's discipline design variable x_3 under
# the constraint g_3.

propulsion_sc = MDOScenario(
    sub_disciplines,
    "y_4",
    design_space.filter(["x_3"], copy=True),
    formulation_settings_model=sub_scenario_settings,
    maximize_objective=True,
    name="PropulsionScenario",
)
propulsion_sc.set_algorithm(algo_settings_model=sc_algo_settings)
propulsion_sc.formulation.optimization_problem.objective *= 0.001
propulsion_sc.add_constraint("g_3", constraint_type="ineq")

# %%
# Build the Aerodynamics Sub-scenario
# -----------------------------------
# This sub-scenario will optimize the aerodynamics' discipline design variable x_2 under
# the constraint g_2.

aerodynamics_sc = MDOScenario(
    sub_disciplines,
    "y_4",
    design_space.filter(["x_2"], copy=True),
    formulation_settings_model=sub_scenario_settings,
    maximize_objective=True,
    name="AerodynamicsScenario",
)
aerodynamics_sc.set_algorithm(algo_settings_model=sc_algo_settings)
aerodynamics_sc.formulation.optimization_problem.objective *= 0.001
aerodynamics_sc.add_constraint("g_2", constraint_type="ineq")

# %%
# Build the Structure Sub-scenario
# -----------------------------------
# This sub-scenario will optimize the structure's discipline design variable x_1 under
# the constraint g_1.

structure_sc = MDOScenario(
    sub_disciplines,
    "y_4",
    design_space.filter(["x_1"], copy=True),
    formulation_settings_model=sub_scenario_settings,
    maximize_objective=True,
    name="StructureScenario",
)
structure_sc.set_algorithm(algo_settings_model=sc_algo_settings)
structure_sc.formulation.optimization_problem.objective *= 0.001
structure_sc.add_constraint("g_1", constraint_type="ineq")

# %%
# System's Scenario Settings
# ---------------------------
# The BiLevel BCD formulation allows to independently define the settings
# for the BCD MDA, such as shown below.

bcd_mda_settings = MDAGaussSeidel_Settings(tolerance=1e-5, max_mda_iter=10)

# %%
# Then, you may pass the BCD MDA settings directly to the formulation settings.

system_settings = BiLevel_BCD_Settings(
    bcd_mda_settings=bcd_mda_settings,
)

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
# Just like for the sub-scenario, we define the algorithm settings for the
# system scenario.

system_sc_algo_settings = NLOPT_COBYLA_Settings(max_iter=100)

# %%
# Build the System's Scenario
# -----------------------------------
# The system level scenario is based on the three previous sub-scenarios for which we aim to maximize the range.

sub_scenarios = [propulsion_sc, aerodynamics_sc, structure_sc, mission_disc]

system_scenario = MDOScenario(
    sub_scenarios,
    "y_4",
    design_space.filter(["x_shared"], copy=True),
    formulation_settings_model=system_settings,
    maximize_objective=True,
)
system_scenario.formulation.optimization_problem.objective *= 0.001
system_scenario.set_algorithm(algo_settings_model=system_sc_algo_settings)
system_scenario.add_constraint("g_1", constraint_type="ineq")
system_scenario.add_constraint("g_2", constraint_type="ineq")
system_scenario.add_constraint("g_3", constraint_type="ineq")


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
system_scenario.execute()

# %%
# Plot the history of the MDA residuals
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# For the first MDA:
system_scenario.formulation.mda1.plot_residual_history(save=False, show=True)

# %%
# For the second MDA:
system_scenario.formulation.mda2.plot_residual_history(save=False, show=True)

# %%
# For the BCD MDA:
system_scenario.formulation.bcd_mda.plot_residual_history(save=False, show=True)

# %%
# Plot the system optimization history view
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
system_scenario.post_process(post_name="OptHistoryView", save=False, show=True)

# %%
# Plot the structure optimization histories of the 2 first iterations
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
struct_databases = system_scenario.formulation.scenario_adapters[2].databases
for database in struct_databases[:2]:
    opt_problem = deepcopy(structure_sc.formulation.optimization_problem)
    opt_problem.database = database
    execute_post(opt_problem, post_name="OptHistoryView", save=False, show=True)

# %%
# Print execution metrics on disciplines and sub-scenarios
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

for disc in [propulsion_disc, aerodynamics_disc, mission_disc, structure_disc]:
    print(f"{disc.name}: {disc.execution_statistics.n_executions} calls.")

for sub_sc in [propulsion_sc, aerodynamics_sc, structure_sc]:
    print(f"{sub_sc.name}: {sub_sc.execution_statistics.n_executions} calls.")
