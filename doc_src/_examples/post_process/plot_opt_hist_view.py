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
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Optimization History View
=========================

In this example, we illustrate the use of the :class:`.OptHistoryView` plot
on the Sobieski's SSBJ problem.
"""
from __future__ import annotations

from gemseo.api import configure_logger
from gemseo.api import create_discipline
from gemseo.api import create_scenario
from gemseo.problems.sobieski.core.problem import SobieskiProblem

###############################################################################
# Import
# ------
# The first step is to import some functions from the API
# and a method to get the design space.

configure_logger()
###############################################################################
# Description
# -----------
# The **OptHistoryView** post-processing
# creates a series of plots:
#
# - The design variables history - This graph shows the normalized values of the
#   design variables, the :math:`y` axis is the index of the inputs in the vector;
#   and the :math:`x` axis represents the iterations.
# - The objective function history - It shows the evolution of the objective
#   value during the optimization.
# - The distance to the best design variables - Plots the vector
#   :math:`log( ||x-x^*|| )` in log scale.
# - The history of the Hessian approximation of the objective - Plots an approximation
#   of the second order derivatives of the objective function
#   :math:`\frac{\partial^2 f(x)}{\partial x^2}`, which is a measure of
#   the sensitivity of the function with respect to the design variables,
#   and of the anisotropy of the problem (differences of curvatures in the
#   design space).
# - The inequality constraint history - Portrays the evolution of the values of the
#   :term:`constraints`. The inequality constraints must be non-positive, that is why
#   the plot must be green or white for satisfied constraints (white = active,
#   red = violated). For an :ref:`IDF formulation <idf_formulation>`, an additional
#   plot is created to track the equality constraint history.

###############################################################################
# Create disciplines
# ------------------
# At this point we instantiate the disciplines of Sobieski's SSBJ problem:
# Propulsion, Aerodynamics, Structure and Mission
disciplines = create_discipline(
    [
        "SobieskiPropulsion",
        "SobieskiAerodynamics",
        "SobieskiStructure",
        "SobieskiMission",
    ]
)

###############################################################################
# Create design space
# -------------------
# We also read the design space from the :class:`.SobieskiProblem`.
design_space = SobieskiProblem().design_space

###############################################################################
# Create and execute scenario
# ---------------------------
# The next step is to build an MDO scenario in order to maximize the range,
# encoded 'y_4', with respect to the design parameters, while satisfying the
# inequality constraints 'g_1', 'g_2' and 'g_3'. We can use the MDF formulation,
# the SLSQP optimization algorithm
# and a maximum number of iterations equal to 100.
scenario = create_scenario(
    disciplines,
    formulation="MDF",
    objective_name="y_4",
    maximize_objective=True,
    design_space=design_space,
)
scenario.set_differentiation_method()
for constraint in ["g_1", "g_2", "g_3"]:
    scenario.add_constraint(constraint, "ineq")
scenario.execute({"algo": "SLSQP", "max_iter": 10})

###############################################################################
# Post-process scenario
# ---------------------
# Lastly, we post-process the scenario by means of the :class:`.OptHistoryView`
# plot which plots the history of optimization for both objective function,
# constraints, design parameters and distance to the optimum.

###############################################################################
# .. tip::
#
#    Each post-processing method requires different inputs and offers a variety
#    of customization options. Use the API function
#    :meth:`~gemseo.api.get_post_processing_options_schema` to print a table with
#    the options for any post-processing algorithm.
#    Or refer to our dedicated page:
#    :ref:`gen_post_algos`.
scenario.post_process("OptHistoryView", save=False, show=True)
