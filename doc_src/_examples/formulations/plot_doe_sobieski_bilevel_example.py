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
BiLevel-based DOE on the Sobieski SSBJ test case
================================================
"""

from __future__ import annotations

from copy import deepcopy
from os import name as os_name

from gemseo import configure_logger
from gemseo import create_discipline
from gemseo import create_scenario
from gemseo import execute_post
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
# Then, we build a sub-scenario for each strongly coupled disciplines.

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

# %%
# Build a sub-scenario for Structure
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# This sub-scenario will maximize
# log(aircraft total weight / (aircraft total weight - fuel weight)).
sc_str = create_scenario(
    struct,
    "y_11",
    deepcopy(design_space).filter("x_1"),
    name="StructureScenario",
    maximize_objective=True,
    formulation_name="DisciplinaryOpt",
)

# %%
# Build a scenario for Mission
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# This scenario is based on the three previous sub-scenarios and on the
# Mission and aims to maximize the range (Breguet).
sub_disciplines = [sc_prop, sc_aero, sc_str, mission]
system_scenario = create_scenario(
    sub_disciplines,
    "y_4",
    design_space.filter("x_shared", copy=True),
    parallel_scenarios=False,
    reset_x0_before_opt=True,
    scenario_type="DOE",
    formulation_name="BiLevel",
    save_opt_history="True",
    naming="UUID",
)

# %%
# .. note::
#
#    Setting ``reset_x0_before_opt=True`` is mandatory when doing a DOE
#    in parallel. If we want reproducible results, don't reuse previous xopt.

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
#    databases to the disk using ``save_opt_history=True``. If your sub-scenarios are
#    running in parallel, and you are saving the optimization histories to the disk, set
#    the ``naming`` setting to ``"UUID"``, which is multiprocessing-safe.
#    The setting ``keep_opt_history`` will not work if the sub-scenarios are running in
#    parallel because the databases are not copied from the sub-processes to the main
#    process. In this case you shall always save the optimization history to the disk.

system_scenario.formulation.mda1.warm_start = False
system_scenario.formulation.mda2.warm_start = False

# %%
# .. note::
#
#    This is mandatory when doing a DOE in parallel if we want always exactly
#    the same results, don't warm start mda1 to have exactly the same
#    process whatever the execution order and process dispatch.

for sub_sc in sub_disciplines[0:3]:
    sub_sc.set_algorithm(algo_name="L-BFGS-B", max_iter=20)

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
# Multiprocessing
# ^^^^^^^^^^^^^^^
# It is possible to run a DOE in parallel using multiprocessing, in order to do
# this, we specify the number of processes to be used for the computation of
# the samples.

# %%
# .. warning::
#    The multiprocessing option has some limitations on Windows.
#    Due to problems with sphinx, we disable it in this example.
#    The features :class:`.MemoryFullCache` and :class:`.HDF5Cache` are not
#    available for multiprocessing on Windows.
#    As an alternative, we recommend the method
#    :meth:`.DOEScenario.set_optimization_history_backup`.
n_processes = 1 if os_name == "nt" else 4
system_scenario.execute(algo_name="PYDOE_LHS", n_samples=30, n_processes=n_processes)

system_scenario.print_execution_metrics()

# %%
# .. warning::
#    On Windows, the progress bar may show duplicated instances during the
#    initialization of each subprocess. In some cases it may also print the
#    conclusion of an iteration ahead of another one that was concluded first.
#    This is a consequence of the pickling process and does not affect the
#    computations of the scenario.

# %%
# Exporting the problem data.
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# After the execution of the scenario, you may want to export your data to use it
# elsewhere. The method :meth:`.Scenario.to_dataset` will allow you to export
# your results to a :class:`.Dataset`, the basic |g| class to store data.
dataset = system_scenario.to_dataset("a_name_for_my_dataset")

# %%
# Plot the optimization history view
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
system_scenario.post_process(post_name="OptHistoryView", save=False, show=True)

# %%
# Plot the scatter matrix
# ^^^^^^^^^^^^^^^^^^^^^^^
system_scenario.post_process(
    post_name="ScatterPlotMatrix",
    variable_names=["y_4", "x_shared"],
    save=False,
    show=True,
)

# %%
# Plot parallel coordinates
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
system_scenario.post_process(post_name="ParallelCoordinates", save=False, show=True)

# %%
# Plot correlations
# ^^^^^^^^^^^^^^^^^
system_scenario.post_process(post_name="Correlations", save=False, show=True)

# %%
# Plot the structure optimization histories of the 2 first iterations
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The code below will not work if you ran the system scenario with ``n_processes`` > 1.
# Indeed, parallel execution of sub-scenarios prevents us to save the databases from
# each sub-process to the main process. If you ran the system scenario with many
# processes, you can still save the databases to the disk with
# ``save_opt_history=True`` and ``naming="UUID"``. Refer to the formulation settings for
# more information.
struct_databases = system_scenario.formulation.scenario_adapters[2].databases
for database in struct_databases[:2]:
    opt_problem = deepcopy(sc_str.formulation.optimization_problem)
    opt_problem.database = database
    execute_post(opt_problem, post_name="OptHistoryView", save=False, show=True)
