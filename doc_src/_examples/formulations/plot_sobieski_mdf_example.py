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
"""MDF-based MDO on the Sobieski SSBJ test case.
=============================================
"""

from __future__ import annotations

from gemseo import configure_logger
from gemseo import create_discipline
from gemseo import create_scenario
from gemseo import generate_n2_plot
from gemseo.problems.mdo.sobieski.core.design_space import SobieskiDesignSpace

configure_logger()

# %%
# Instantiate the disciplines
# ---------------------------
# First, we instantiate the four disciplines of the use case:
# :class:`.SobieskiPropulsion`,
# :class:`.SobieskiAerodynamics`,
# :class:`.SobieskiMission`
# and :class:`.SobieskiStructure`.
disciplines = create_discipline([
    "SobieskiPropulsion",
    "SobieskiAerodynamics",
    "SobieskiMission",
    "SobieskiStructure",
])

# %%
# We can quickly access the most relevant information of any discipline (name, inputs,
# and outputs) with Python's ``print()`` function. Moreover, we can get the default
# input values of a discipline with the attribute :attr:`.Discipline.default_input_data`
for discipline in disciplines:
    print(discipline)  # noqa: T201
    print(f"Default inputs: {discipline.default_input_data}")  # noqa: T201


# %%
# You may also be interested in plotting the couplings of your disciplines.
# A quick way of getting this information is the high-level function
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
# MDF formulations. The MDF formulation includes an MDA, and thus one of the settings of
# the formulation is ``main_mda_settings``, which configures the solver for the strong
# couplings.
main_mda_settings = {
    "tolerance": 1e-14,
    "max_mda_iter": 50,
    "warm_start": True,
    "use_lu_fact": False,
    "linear_solver_tolerance": 1e-14,
}

# %%
#
# - ``'warm_start``: warm starts MDA,
# - ``'use_lu_fact``: optimize the adjoint resolution by storing
#   the Jacobian matrix LU factorization for the multiple RHS
#   (objective + constraints). This saves CPU time if you can pay for
#   the memory and have the full Jacobians available, not just matrix vector
#   products.
# - ``'linear_solver_tolerance'``: set the linear solver tolerance,
#   idem we need full convergence
#
design_space = SobieskiDesignSpace()
design_space

# %%
scenario = create_scenario(
    disciplines,
    "y_4",
    design_space,
    maximize_objective=True,
    formulation_name="MDF",
    main_mda_settings=main_mda_settings,
)

# %%
# Set the design constraints
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
for c_name in ["g_1", "g_2", "g_3"]:
    scenario.add_constraint(c_name, constraint_type="ineq")

# %%
# XDSMIZE the scenario
# ^^^^^^^^^^^^^^^^^^^^
# Generate the XDSM file on the fly:
#
# - ``log_workflow_status=True`` will log the status of the workflow  in the console,
# - ``save_html`` (default ``True``) will generate a self-contained HTML file,
#   that can be automatically opened using ``show_html=True``.
scenario.xdsmize(save_html=False, pdf_build=False)

# %%
# Define the algorithm inputs
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We set the maximum number of iterations, the optimizer
# and the optimizer options. Algorithm specific options are passed there.
# Use the high-level function :func:`.get_algorithm_options_schema`
# for more information or read the documentation.
#
# Here the ``ftol_rel`` setting is a stop criteria based on the relative difference
# in the objective between two iterates ineq_tolerance the tolerance
# determination of the optimum; this is specific to the |g| wrapping and not
# in the solver.
from gemseo.settings.opt import SLSQP_Settings  # noqa: E402

slsqp_settings = SLSQP_Settings(
    max_iter=10,
    ftol_rel=1e-10,
    ineq_tolerance=2e-3,
    normalize_design_space=True,
)

# %%
# .. seealso::
#
#    We can also generate a backup file for the optimization,
#    as well as plots on the fly of the optimization history if option
#    ``plot`` is ``True``.
#    This slows down a lot the process, here since SSBJ is very light
#
#    .. code::
#
#     scenario.set_optimization_history_backup(
#         file_path="mdf_backup.h5",
#         at_each_iteration=True,
#         at_each_function_call=False,
#         erase=True,
#         load=False,
#         plot=True
#     )

# %%
# Execute the scenario
# ^^^^^^^^^^^^^^^^^^^^
scenario.execute(slsqp_settings)

# %%
# Save the optimization history
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We can save the whole optimization problem and its history for further
# post-processing:
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
scenario.post_process(post_name="OptHistoryView", save=False, show=True)

# %%
# Note that post-processor settings passed to :class:`.BaseScenario.post_process` can be
# provided via a Pydantic model (see the example below). For more information,
# see :ref:`post_processor_settings`.

# %%
# Plot the basic history view
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
from gemseo.settings.post import BasicHistory_Settings  # noqa: E402

scenario.post_process(
    BasicHistory_Settings(variable_names=["x_shared"], save=False, show=True)
)

# %%
# Plot the constraints and objective history
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
scenario.post_process(post_name="ObjConstrHist", save=False, show=True)

# %%
# Plot the constraints history
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
scenario.post_process(
    post_name="ConstraintsHistory",
    constraint_names=["g_1", "g_2", "g_3"],
    save=False,
    show=True,
)

# %%
# Plot the constraints history using a radar chart
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
scenario.post_process(
    post_name="RadarChart",
    constraint_names=["g_1", "g_2", "g_3"],
    save=False,
    show=True,
)

# %%
# Plot the quadratic approximation of the objective
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
scenario.post_process(post_name="QuadApprox", function="-y_4", save=False, show=True)

# %%
# Plot the functions using a SOM
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
scenario.post_process(post_name="SOM", save=False, show=True)

# %%
# Plot the scatter matrix of variables of interest
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
scenario.post_process(
    post_name="ScatterPlotMatrix",
    variable_names=["-y_4", "g_1"],
    save=False,
    show=True,
    fig_size=(14, 14),
)

# %%
# Plot the variables using the parallel coordinates
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
scenario.post_process(post_name="ParallelCoordinates", save=False, show=True)

# %%
# Plot the robustness of the solution
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
scenario.post_process(post_name="Robustness", save=False, show=True)

# %%
# Plot the influence of the design variables
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
scenario.post_process(
    post_name="VariableInfluence", fig_size=(14, 14), save=False, show=True
)
