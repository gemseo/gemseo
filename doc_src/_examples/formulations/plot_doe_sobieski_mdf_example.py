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
MDF-based DOE on the Sobieski SSBJ test case
============================================
"""

from __future__ import annotations

from os import name as os_name

from gemseo import configure_logger
from gemseo import create_discipline
from gemseo import create_scenario
from gemseo import generate_n2_plot
from gemseo.problems.mdo.sobieski.core.design_space import SobieskiDesignSpace
from gemseo.settings.doe import PYDOE_LHS_Settings

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
# A quick way of getting this information is the high-level function
# :func:`.generate_n2_plot`. A much more detailed explanation of coupling
# visualization is available :ref:`here <coupling_visualization>`.
generate_n2_plot(disciplines, save=False, show=True)

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
design_space

# %%
# Instantiate the scenario
# ^^^^^^^^^^^^^^^^^^^^^^^^
scenario = create_scenario(
    disciplines,
    "y_4",
    design_space,
    maximize_objective=True,
    scenario_type="DOE",
    formulation_name="MDF",
)

# %%
# Set the design constraints
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
for constraint in ["g_1", "g_2", "g_3"]:
    scenario.add_constraint(constraint, constraint_type="ineq")

# %%
# Visualize the XDSM
# ^^^^^^^^^^^^^^^^^^
# Generate the XDSM on the fly:
#
# - ``log_workflow_status=True`` will log the status of the workflow in the console,
# - ``save_html`` (default ``True``) will generate a self-contained HTML file,
#   that can be automatically opened using ``show_html=True``.
scenario.xdsmize(save_html=False)

# %%
# Execute the scenario
# ^^^^^^^^^^^^^^^^^^^^
# Use provided analytic derivatives
scenario.set_differentiation_method()

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

# %%
# We define the algorithm settings. Here the criterion "center" of pyDOE
# centers the points within the sampling intervals.
# Note that it is also possible to pass the settings one by one, see
# :ref:`algorithm_settings`.

lhs_settings = PYDOE_LHS_Settings(
    n_samples=30,
    criterion="center",
    # Evaluate gradient of the MDA
    # with coupled adjoint
    eval_jac=True,
    # Run in parallel on 1 or 4 processors
    n_processes=1 if os_name == "nt" else 4,
)

scenario.execute(lhs_settings)

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
dataset = scenario.to_dataset("a_name_for_my_dataset")

# %%
# Plot the optimization history view
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
scenario.post_process(post_name="OptHistoryView", save=False, show=True)

# %%
# .. tip::
#
#    Each post-processing method requires different inputs and offers a variety
#    of customization options. Use the high-level function
#    :func:`.get_post_processing_options_schema` to print a table with
#    the attributes for any post-processing algo. Or refer to our dedicated page:
#    :ref:`gen_post_algos`.

# %%
# Plot the scatter matrix
# ^^^^^^^^^^^^^^^^^^^^^^^
scenario.post_process(
    post_name="ScatterPlotMatrix",
    save=False,
    show=True,
    variable_names=["y_4", "x_shared"],
)

# %%
# Plot correlations
# ^^^^^^^^^^^^^^^^^
scenario.post_process(post_name="Correlations", save=False, show=True)
