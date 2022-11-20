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
Basic history
=============

In this example, we illustrate the use of the :class:`.BasicHistory` plot
on the Sobieski's SSBJ problem.
"""
from __future__ import annotations

from gemseo.api import configure_logger
from gemseo.api import create_discipline
from gemseo.api import create_scenario
from gemseo.problems.sobieski.core.problem import SobieskiProblem

# %%
# Import
# ------
# The first step is to import some functions from the API
# and a method to get the design space.

configure_logger()

# %%
# Description
# -----------
# The :class:`~gemseo.post.basic_history.BasicHistory` post-processing
# plots any of the constraint or objective functions
# w.r.t. the optimization iterations or sampling snapshots.

# %%
# Create disciplines
# ------------------
# At this point, we instantiate the disciplines of Sobieski's SSBJ problem:
# Propulsion, Aerodynamics, Structure and Mission
disciplines = create_discipline(
    [
        "SobieskiPropulsion",
        "SobieskiAerodynamics",
        "SobieskiStructure",
        "SobieskiMission",
    ]
)

# %%
# Create design space
# -------------------
# We also read the design space from the :class:`.SobieskiProblem`.
design_space = SobieskiProblem().design_space

# %%
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

# %%
# Post-process scenario
# ---------------------
# Lastly, we post-process the scenario by means of the :class:`.BasicHistory`
# plot which plots any of the constraint or objective functions
# w.r.t. optimization iterations or sampling snapshots.

# %%
# .. tip::
#
#    Each post-processing method requires different inputs and offers a variety
#    of customization options. Use the API function
#    :meth:`~gemseo.api.get_post_processing_options_schema` to print a table with
#    the options for any post-processing algorithm.
#    Or refer to our dedicated page:
#    :ref:`gen_post_algos`.
scenario.post_process(
    "BasicHistory",
    variable_names=["g_1", "g_2", "g_3"],
    save=False,
    show=True,
)
scenario.post_process("BasicHistory", variable_names=["y_4"], save=False, show=True)
# %%
# .. note::
#
#    Set the boolean instance attribute
#    :attr:`.OptimizationProblem.use_standardized_objective` to ``False``
#    to plot the objective to maximize as a performance function.
scenario.use_standardized_objective = False
scenario.post_process("BasicHistory", variable_names=["y_4"], save=False, show=True)
