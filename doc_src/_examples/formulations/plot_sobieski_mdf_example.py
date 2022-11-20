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
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
MDF-based MDO on the Sobieski SSBJ test case
============================================
"""
from __future__ import annotations

from gemseo.api import configure_logger
from gemseo.api import create_discipline
from gemseo.api import create_scenario
from gemseo.api import generate_n2_plot
from gemseo.problems.sobieski.core.problem import SobieskiProblem

configure_logger()

# %%
# Instantiate the disciplines
# ---------------------------
# First, we instantiate the four disciplines of the use case:
# :class:`.SobieskiPropulsion`,
# :class:`.SobieskiAerodynamics`,
# :class:`.SobieskiMission`
# and :class:`.SobieskiStructure`.
disciplines = create_discipline(
    [
        "SobieskiPropulsion",
        "SobieskiAerodynamics",
        "SobieskiMission",
        "SobieskiStructure",
    ]
)

# %%
# We can quickly access the most relevant information of any discipline (name, inputs,
# and outputs) with Python's ``print()`` function. Moreover, we can get the default
# input values of a discipline with the attribute :attr:`.MDODiscipline.default_inputs`
for discipline in disciplines:
    print(discipline)
    print(f"Default inputs: {discipline.default_inputs}")

# %%
# You may also be interested in plotting the couplings of your disciplines.
# A quick way of getting this information is the API function
# :func:`.generate_n2_plot`. A much more detailed explanation of coupling
# visualization is available :ref:`here <coupling_visualization>`.
generate_n2_plot(disciplines, save=False, show=True)

# %%
# Build, execute and post-process the scenario
# --------------------------------------------
# Then, we build the scenario which links the disciplines
# with the formulation and the optimization algorithm. Here, we use the
# :class:`.MDF` formulation. We tell the scenario to minimize -y_4 instead of
# minimizing y_4 (range), which is the default option.
#
# Instantiate the scenario
# ^^^^^^^^^^^^^^^^^^^^^^^^
# During the instantiation of the scenario, we provide some options for the
# MDF formulations:
formulation_options = {
    "tolerance": 1e-10,
    "max_mda_iter": 50,
    "warm_start": True,
    "use_lu_fact": True,
    "linear_solver_tolerance": 1e-15,
}

# %%
#
# - :code:`'warm_start`: warm starts MDA,
# - :code:`'warm_start`: optimize the adjoints resolution by storing
#   the Jacobian matrix LU factorization for the multiple RHS
#   (objective + constraints). This saves CPU time if you can pay for
#   the memory and have the full Jacobians available, not just matrix vector
#   products.
# - :code:`'linear_solver_tolerance'`: set the linear solver tolerance,
#   idem we need full convergence
#
design_space = SobieskiProblem().design_space
print(design_space)
scenario = create_scenario(
    disciplines,
    "MDF",
    objective_name="y_4",
    design_space=design_space,
    maximize_objective=True,
    **formulation_options,
)

# %%
# Set the design constraints
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
for c_name in ["g_1", "g_2", "g_3"]:
    scenario.add_constraint(c_name, "ineq")

# %%
# XDSMIZE the scenario
# ^^^^^^^^^^^^^^^^^^^^
# Generate the XDSM file on the fly, setting ``print_statuses=True``
# will print the status in the console
# ``html_output`` (default ``True``), will generate a self-contained
# HTML file, that can be automatically open using ``open_browser=True``
scenario.xdsmize()

# %%
# Define the algorithm inputs
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We set the maximum number of iterations, the optimizer
# and the optimizer options. Algorithm specific options are passed there.
# Use :meth:`~gemseo.api.get_algorithm_options_schema` API function for more
# information or read the documentation.
#
# Here ftol_rel option is a stop criteria based on the relative difference
# in the objective between two iterates ineq_tolerance the tolerance
# determination of the optimum; this is specific to the |g| wrapping and not
# in the solver.
algo_options = {
    "ftol_rel": 1e-10,
    "ineq_tolerance": 2e-3,
    "normalize_design_space": True,
}
scn_inputs = {"max_iter": 10, "algo": "SLSQP", "algo_options": algo_options}

# %%
# .. seealso::
#
#    We can also generate a backup file for the optimization,
#    as well as plots on the fly of the optimization history if option
#    ``generate_opt_plot`` is ``True``.
#    This slows down a lot the process, here since SSBJ is very light
#
#    .. code::
#
#     scenario.set_optimization_history_backup(file_path="mdf_backup.h5",
#                                              each_new_iter=True,
#                                              each_store=False, erase=True,
#                                              pre_load=False,
#                                              generate_opt_plot=True)

# %%
# Execute the scenario
# ^^^^^^^^^^^^^^^^^^^^
scenario.execute(scn_inputs)

# %%
# Save the optimization history
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We can save the whole optimization problem and its history for further post
# processing:
scenario.save_optimization_history("mdf_history.h5", file_format="hdf5")

# %%
# We can also save only calls to functions and design variables history:
scenario.save_optimization_history("mdf_history.xml", file_format="ggobi")

# %%
# Print optimization metrics
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
scenario.print_execution_metrics()

# %%
# Post-process the results
# ------------------------
#
# Plot the optimization history view
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
scenario.post_process("OptHistoryView", save=False, show=True)

# %%
# Plot the basic history view
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
scenario.post_process(
    "BasicHistory", variable_names=["x_shared"], save=False, show=True
)

# %%
# Plot the constraints and objective history
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
scenario.post_process("ObjConstrHist", save=False, show=True)

# %%
# Plot the constraints history
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
scenario.post_process(
    "ConstraintsHistory",
    constraint_names=["g_1", "g_2", "g_3"],
    save=False,
    show=True,
)

# %%
# Plot the constraints history using a radar chart
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
scenario.post_process(
    "RadarChart",
    constraint_names=["g_1", "g_2", "g_3"],
    save=False,
    show=True,
)

# %%
# Plot the quadratic approximation of the objective
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
scenario.post_process("QuadApprox", function="-y_4", save=False, show=True)

# %%
# Plot the functions using a SOM
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
scenario.post_process("SOM", save=False, show=True)

# %%
# Plot the scatter matrix of variables of interest
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
scenario.post_process(
    "ScatterPlotMatrix",
    variable_names=["-y_4", "g_1"],
    save=False,
    show=True,
    fig_size=(14, 14),
)

# %%
# Plot the variables using the parallel coordinates
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
scenario.post_process("ParallelCoordinates", save=False, show=True)

# %%
# Plot the robustness of the solution
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
scenario.post_process("Robustness", save=True, show=True)

# %%
# Plot the influence of the design variables
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
scenario.post_process("VariableInfluence", fig_size=(14, 14), save=False, show=True)
