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
IDF-based MDO on the Sobieski SSBJ test case
============================================
"""

from __future__ import annotations

from gemseo import configure_logger
from gemseo import create_discipline
from gemseo import create_scenario
from gemseo import generate_n2_plot
from gemseo.problems.mdo.sobieski.core.design_space import SobieskiDesignSpace
from gemseo.settings.opt import SLSQP_Settings

configure_logger()

# %%
# Instantiate the  disciplines
# ----------------------------
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
    print(discipline)
    print(f"Default inputs: {discipline.default_input_data}")

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
# :class:`.IDF` formulation. We tell the scenario to minimize -y_4 instead of
# minimizing y_4 (range), which is the default option.
#
# Instantiate the scenario
# ^^^^^^^^^^^^^^^^^^^^^^^^
design_space = SobieskiDesignSpace()
design_space

# %%
scenario = create_scenario(
    disciplines,
    "y_4",
    design_space,
    maximize_objective=True,
    formulation_name="IDF",
)

# %%
# Set the design constraints
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
for c_name in ["g_1", "g_2", "g_3"]:
    scenario.add_constraint(c_name, constraint_type="ineq")

# %%
# Visualize the XDSM
# ^^^^^^^^^^^^^^^^^^
# Generate the XDSM file on the fly:
#
# - ``log_workflow_status=True`` will log the status of the workflow  in the console,
# - ``save_html`` (default ``True``) will generate a self-contained HTML file,
#   that can be automatically opened using ``show_html=True``.
scenario.xdsmize(save_html=False)

# %%
# Define the algorithm inputs
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We set the maximum number of iterations, the optimizer
# and the optimizer settings

slsqp_settings = SLSQP_Settings(
    max_iter=20,
    ftol_rel=1e-10,
    ineq_tolerance=1e-3,
    eq_tolerance=1e-3,
    normalize_design_space=True,
)


# %%
# Execute the scenario
# ^^^^^^^^^^^^^^^^^^^^
scenario.execute(slsqp_settings)

# %%
# Save the optimization history
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We can save the whole optimization problem and its history for further post
# processing:
scenario.save_optimization_history("idf_history.h5", file_format="hdf5")

# %%
# We can also save only calls to functions and design variables history:
scenario.save_optimization_history("idf_history.xml", file_format="ggobi")

# %%
# Print optimization metrics
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
scenario.print_execution_metrics()

# %%
# Plot the optimization history view
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
scenario.post_process(post_name="OptHistoryView", save=False, show=True)

# %%
# Plot the quadratic approximation of the objective
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
scenario.post_process(post_name="QuadApprox", function="-y_4", save=False, show=True)
