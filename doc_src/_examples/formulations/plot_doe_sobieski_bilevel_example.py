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

from gemseo.api import configure_logger
from gemseo.api import create_discipline
from gemseo.api import create_scenario
from gemseo.problems.sobieski.core.problem import SobieskiProblem

configure_logger()

# %%
# Instantiate the  disciplines
# ----------------------------
# First, we instantiate the four disciplines of the use case:
# :class:`.SobieskiPropulsion`,
# :class:`.SobieskiAerodynamics`,
# :class:`.SobieskiMission`
# and :class:`.SobieskiStructure`.
propu, aero, mission, struct = create_discipline(
    [
        "SobieskiPropulsion",
        "SobieskiAerodynamics",
        "SobieskiMission",
        "SobieskiStructure",
    ]
)

# %%
# Build, execute and post-process the scenario
# --------------------------------------------
# Then, we build the scenario which links the disciplines
# with the formulation and the optimization algorithm. Here, we use the
# :class:`.BiLevel` formulation. We tell the scenario to minimize -y_4
# instead of minimizing y_4 (range), which is the default option.
#
# We need to define the design space.
design_space = SobieskiProblem().design_space

# %%
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

# %%
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

# %%
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

# %%
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

# %%
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
    parallel_scenarios=False,
    reset_x0_before_opt=True,
    scenario_type="DOE",
)

# %%
# .. note::
#
#    Setting :code:`reset_x0_before_opt=True` is mandatory when doing a DOE
#    in parallel. If we want reproducible results, don't reuse previous xopt.

system_scenario.formulation.mda1.warm_start = False
system_scenario.formulation.mda2.warm_start = False

# %%
# .. note::
#
#    This is mandatory when doing a DOE in parallel if we want always exactly
#    the same results, don't warm start mda1 to have exactly the same
#    process whatever the execution order and process dispatch.

for sub_sc in sub_disciplines[0:3]:
    sub_sc.default_inputs = {"max_iter": 20, "algo": "L-BFGS-B"}

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
#    For Python versions < 3.7 and Numpy < 1.20.0, subprocesses may get hung
#    randomly during execution. It is strongly recommended to update your
#    environment to avoid this problem.
#    The features :class:`.MemoryFullCache` and :class:`.HDF5Cache` are not
#    available for multiprocessing on Windows.
#    As an alternative, we recommend the method
#    :meth:`.DOEScenario.set_optimization_history_backup`.
system_scenario.execute(
    {
        "n_samples": 30,
        "algo": "lhs",
        "algo_options": {"n_processes": 1 if os_name == "nt" else 4},
    }
)

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
# elsewhere. The method :meth:`.Scenario.export_to_dataset` will allow you to export
# your results to a :class:`.Dataset`, the basic |g| class to store data.
# From a dataset, you can even obtain a Pandas dataframe with the method
# :meth:`~.Dataset.export_to_dataframe`:
dataset = system_scenario.export_to_dataset("a_name_for_my_dataset")
dataframe = dataset.export_to_dataframe()

# %%
# Plot the optimization history view
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
system_scenario.post_process("OptHistoryView", save=False, show=True)

# %%
# Plot the scatter matrix
# ^^^^^^^^^^^^^^^^^^^^^^^
system_scenario.post_process(
    "ScatterPlotMatrix",
    variable_names=["y_4", "x_shared"],
    save=False,
    show=True,
)

# %%
# Plot parallel coordinates
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
system_scenario.post_process("ParallelCoordinates", save=False, show=True)

# %%
# Plot correlations
# ^^^^^^^^^^^^^^^^^
system_scenario.post_process("Correlations", save=False, show=True)
