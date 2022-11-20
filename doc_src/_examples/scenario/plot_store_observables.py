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
#        :author: Gilberto Ruiz Jimenez
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Store observables
=================
"""
# %%
# Introduction
# ------------
# In this example,
# we will learn how to store the history of state variables using the
# :meth:`~gemseo.core.scenario.Scenario.add_observable` method.
# This is useful in situations where we wish to access, post-process,
# or save the values of discipline outputs that are not design variables,
# constraints or objective functions.
#
# The Sellar problem
# ------------------
# We will consider in this example the Sellar problem:
#
# .. include:: /tutorials/_description/sellar_problem_definition.inc
#
# Imports
# -------
# All the imports needed for the tutorials are performed here.
from __future__ import annotations

from gemseo.algos.design_space import DesignSpace
from gemseo.api import configure_logger
from gemseo.api import create_discipline
from gemseo.api import create_scenario
from numpy import array
from numpy import ones

configure_logger()


# %%
# Create the problem disciplines
# ------------------------------
# In this section,
# we use the available classes :class:`.Sellar1`, :class:`.Sellar2`
# and :class:`.SellarSystem` to define the disciplines of the problem.
# The :meth:`~gemseo.api.create_discipline` API function allows us to
# carry out this task easily, as well as store the instances in a list
# to be used later on.
disciplines = create_discipline(["Sellar1", "Sellar2", "SellarSystem"])

# %%
# Create and execute the scenario
# -------------------------------
#
# Create the design space
# ^^^^^^^^^^^^^^^^^^^^^^^
# In this section,
# we define the design space which will be used for the creation of the MDOScenario.
design_space = DesignSpace()
design_space.add_variable("x_local", l_b=0.0, u_b=10.0, value=ones(1))
design_space.add_variable(
    "x_shared", 2, l_b=(-10, 0.0), u_b=(10.0, 10.0), value=array([4.0, 3.0])
)

# %%
# Create the scenario
# ^^^^^^^^^^^^^^^^^^^
# In this section,
# we build the MDO scenario which links the disciplines with the formulation,
# the design space and the objective function.
scenario = create_scenario(
    disciplines, formulation="MDF", objective_name="obj", design_space=design_space
)

# %%
# Add the constraints
# ^^^^^^^^^^^^^^^^^^^
# Then,
# we have to set the design constraints
scenario.add_constraint("c_1", "ineq")
scenario.add_constraint("c_2", "ineq")

# %%
# Add the observables
# ^^^^^^^^^^^^^^^^^^^
# Only the design variables, objective function and constraints are stored by
# default. In order to be able to recover the data from the state variables,
# y1 and y2, we have to add them as observables. All we have to do is enter
# the variable name as a string to the
# :meth:`~gemseo.core.scenario.Scenario.add_observable` method.
# If more than one output name is provided (as a list of strings),
# the observable function returns a concatenated array of the output values.
scenario.add_observable("y_1")
# %%
# It is also possible to add the observable with a custom name,
# using the option `observable_name`. Let us store the variable `y_2` as `y2`.
scenario.add_observable("y_2", observable_name="y2")

# %%
# Execute the scenario
# ^^^^^^^^^^^^^^^^^^^^
# Then,
# we execute the MDO scenario with the inputs of the MDO scenario as a dictionary.
# In this example,
# the gradient-based `SLSQP` optimizer is selected, with 10 iterations at maximum:
scenario.execute(input_data={"max_iter": 10, "algo": "SLSQP"})

# %%
# Access the observable variables
# -------------------------------
# Retrieve observables from a dataset
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# In order to create a dataset, we use the
# corresponding :class:`.OptimizationProblem`:
opt_problem = scenario.formulation.opt_problem
# %%
# We can easily build a dataset from this :class:`.OptimizationProblem`:
# either by separating the design parameters from the functions
# (default option):
dataset = opt_problem.export_to_dataset("sellar_problem")
print(dataset)
# %%
# or by considering all features as default parameters:
dataset = opt_problem.export_to_dataset("sellar_problem", categorize=False)
print(dataset)
# %%
# or by using an input-output naming rather than an optimization naming:
dataset = opt_problem.export_to_dataset("sellar_problem", opt_naming=False)
print(dataset)
# %%
# Access observables by name
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# We can get the observable data by name,
# either as a dictionary indexed by the observable names (default option):
print(dataset.get_data_by_names(["y_1", "y2"]))
# %%
# or as an array:
print(dataset.get_data_by_names(["y_1", "y2"], False))

# %%
# Use the observables in a post-processing method
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Finally,
# we can generate plots with the observable variables. Have a look at the
# Basic History plot and the Scatter Plot Matrix:
scenario.post_process(
    "BasicHistory",
    variable_names=["obj", "y_1", "y2"],
    save=False,
    show=True,
)
scenario.post_process(
    "ScatterPlotMatrix",
    variable_names=["obj", "c_1", "c_2", "y2", "y_1"],
    save=False,
    show=True,
)
