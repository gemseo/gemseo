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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Jean-Christophe Giret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
GEMSEO in 10 minutes
====================
"""
# %%
#
# Introduction
# ------------
#
# This is a short introduction to |g|, geared mainly for new users.  In this
# example, we will set up a simple Multi-disciplinary Design Optimization
# (:term:`MDO`) problem based on a simple analytic problem.
#
# Imports
# -------
#
# First, we will import all the classes and functions needed for the tutorials.
from __future__ import annotations

from math import exp

from gemseo.api import configure_logger
from gemseo.api import create_design_space
from gemseo.api import create_discipline
from gemseo.api import create_scenario
from gemseo.api import generate_n2_plot
from numpy import array
from numpy import ones

# %%
# These imports are needed to compute mathematical expressions and to
# instantiate NumPy arrays. NumPy arrays are used to store numerical data in
# |g| at a low level. If you are not comfortable using NumPy, please have a
# look at the `Numpy Quickstart tutorial
# <https://numpy.org/doc/stable/user/quickstart.html>`_.

# %%
# Here, we configure the |g| logger in order to get information of the process as
# it is executed.

configure_logger()

# %%
#
# A simple MDO test case: the Sellar Problem
# ------------------------------------------
# We will consider in this example the Sellar's problem:
#
# .. include:: /tutorials/_description/sellar_problem_definition.inc
#
# Definition of the disciplines using Python functions
# ----------------------------------------------------
# The Sellar's problem is composed of two :term:`disciplines <discipline>` and
# an :term:`objective function`. As they are expressed analytically, it is
# possible to write them as simple Python functions which take as parameters
# the :term:`design variables` and the :term:`coupling variables`.  The
# returned values may be the outputs of a discipline, the values of the
# :term:`constraints` or the value of the objective function.  Their
# definitions read:


def f_sellar_system(x_local=1.0, x_shared_2=3.0, y_1=1.0, y_2=1.0):
    """Objective function."""
    obj = x_local**2 + x_shared_2 + y_1**2 + exp(-y_2)
    c_1 = 3.16 - y_1**2
    c_2 = y_2 - 24.0
    return obj, c_1, c_2


def f_sellar_1(x_local=1.0, y_2=1.0, x_shared_1=1.0, x_shared_2=3.0):
    """Function for discipline 1."""
    y_1 = (x_shared_1**2 + x_shared_2 + x_local - 0.2 * y_2) ** 0.5
    return y_1


def f_sellar_2(y_1=1.0, x_shared_1=1.0, x_shared_2=3.0):
    """Function for discipline 2."""
    y_2 = abs(y_1) + x_shared_1 + x_shared_2
    return y_2


# %%
# These Python functions can be easily converted into |g|
# :class:`.MDODiscipline` objects by using the :class:`.AutoPyDiscipline`
# discipline. It enables the automatic wrapping of a Python function into a
# |g|
# :class:`.MDODiscipline` by only passing a reference to the function to be
# wrapped. |g| handles the wrapping and the grammar creation under the
# hood. The :class:`.AutoPyDiscipline` discipline can be instantiated using the
# :func:`.create_discipline` function from the |g| :term:`API`:

disc_sellar_system = create_discipline("AutoPyDiscipline", py_func=f_sellar_system)

disc_sellar_1 = create_discipline("AutoPyDiscipline", py_func=f_sellar_1)

disc_sellar_2 = create_discipline("AutoPyDiscipline", py_func=f_sellar_2)

# %%
# Note that it is possible to define the Sellar disciplines by subclassing the
# :class:`.MDODiscipline` class and implementing the constuctor and the _run
# method by hand. Although it would take more time, it may also provide more
# flexibility and more options. This method is illustrated in the :ref:`Sellar
# from scratch tutorial <sellar_from_scratch>`.

# %%
# We then create a list of disciplines, which will be used later to create an
# :class:`.MDOScenario`:
disciplines = [disc_sellar_system, disc_sellar_1, disc_sellar_2]

# %%
# We can quickly access the most relevant information of any discipline (name, inputs,
# and outputs) with Python's ``print()`` function. Moreover, we can get the default
# input values of a discipline with the attribute :attr:`.MDODiscipline.default_inputs`
print(disc_sellar_1)
print(f"Default inputs: {disc_sellar_1.default_inputs}")

# %%
# You may also be interested in plotting the couplings of your disciplines.
# A quick way of getting this information is the API function
# :func:`.generate_n2_plot`. A much more detailed explanation of coupling
# visualization is available :ref:`here <coupling_visualization>`.
generate_n2_plot(disciplines, save=False, show=True)

# %%
# .. note::
#
#    For the sake of clarity, these disciplines are overly simple.
#    Yet, |g| enables the definition of much more complex disciplines,
#    such as wrapping complex :term:`COTS`.
#    Check out the other :ref:`tutorials <tutorials_sg>` and
#    our :ref:`publications list <references>` for more information.

# %%
# Definition of the design space
# ------------------------------
# In order to define :class:`.MDOScenario`,
# a :term:`design space` has to be defined by creating a :class:`.DesignSpace`
# object. The design space definition reads:

design_space = create_design_space()
design_space.add_variable("x_local", l_b=0.0, u_b=10.0, value=ones(1))
design_space.add_variable("x_shared_1", l_b=-10, u_b=10.0, value=array([4.0]))
design_space.add_variable("x_shared_2", l_b=0.0, u_b=10.0, value=array([3.0]))
design_space.add_variable("y_1", l_b=-100.0, u_b=100.0, value=ones(1))
design_space.add_variable("y_2", l_b=-100.0, u_b=100.0, value=ones(1))
print(design_space)

# %%
# Definition of the MDO scenario
# ------------------------------
# Once the disciplines and the design space have been defined,
# we can create our MDO scenario by using the :func:`.create_scenario`
# API call. In this simple example,
# we are using a Multiple Disciplinary Feasible (:term:`MDF`) strategy.
# The Multiple Disciplinary Analyses (:term:`MDA`) are carried out using the
# Gauss-Seidel method. The scenario definition reads:

scenario = create_scenario(
    disciplines,
    formulation="MDF",
    inner_mda_name="MDAGaussSeidel",
    objective_name="obj",
    design_space=design_space,
)

# %%
# It can be noted that neither a :term:`workflow <work flow>`
# nor a :term:`dataflow <data flow>` has been defined.
# By design, there is no need to explicitly define the workflow
# and the dataflow in |g|:
#
# - the workflow is determined from the MDO formulation used.
# - the dataflow is determined from the variable names used in the disciplines.
#   Then, it is of uttermost importance to be consistent while choosing and
#   using the variable names in the disciplines.
#
# .. warning::
#
#    As the workflow and the dataflow are implicitly determined by |g|,
#    set-up errors may easily occur. Although it is not performed
#    in this example, it is strongly advised to
#
#    - check the interfaces between the several disciplines using an N2 diagram,
#    - check the MDO process using an XDSM representation
#
# Setting the constraints
# -----------------------
# Most of the MDO problems are under :term:`constraints`.
# In our problem, we have two inequality constraints,
# and their declaration reads:

scenario.add_constraint("c_1", "ineq")
scenario.add_constraint("c_2", "ineq")

# %%
# Execution of the scenario
# -------------------------
# The scenario is now complete and ready to be executed.
# When running the optimization process,
# the user can choose the optimization algorithm and
# the maximum number of iterations to perform.
# The execution of the scenario reads:

scenario.execute(input_data={"max_iter": 10, "algo": "SLSQP"})

# %%
# The scenario converged after 7 iterations.
# Useful information can be found in the standard output, as seen above.
#
# .. note::
#
#    |g| provides the user with a lot of optimization algorithms and
#    options. An exhaustive list of the algorithms available in |g| can be
#    found in the :ref:`gen_opt_algos` section.

# %%
# Post-processing the results
# ---------------------------
# Post-processings such as plots exhibiting the evolutions of the
# objective function, the design variables or the constraints can be
# extremely useful. The convergence of the objective function, design
# variables and of the inequality constraints can be observed in the
# following plots. Many other post-processings are available in |g| and
# are described in :ref:`Post-processing <post_processing>`.

scenario.post_process("OptHistoryView", save=False, show=True)

# %%
# .. note::
#
#    Such post-processings can be exported in PDF format,
#    by setting :code:`save` to :code:`True` and potentially additional
#    settings (see the :meth:`.Scenario.post_process` options).

# %%
# Exporting the problem data.
# ---------------------------
# After the execution of the scenario, you may want to export your data to use it
# elsewhere. The :meth:`.Scenario.export_to_dataset` will allow you to export your
# results to a :class:`.Dataset`, the basic |g| class to store data.
# From a dataset, you can even obtain a Pandas dataframe with its method
# :meth:`~.Dataset.export_to_dataframe`:
dataset = scenario.export_to_dataset("a_name_for_my_dataset")
dataframe = dataset.export_to_dataframe()

# %%
# What's next?
# ------------
# You have completed a short introduction to |g|.  You can now look at the
# :ref:`tutorials <tutorials_sg>` which exhibit more complex use-cases.  You
# can also have a look at the documentation to discover the several features
# and options of |g|.
