# -*- coding: utf-8 -*-
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
from __future__ import division, unicode_literals

from os import name as os_name

from matplotlib import pyplot as plt

from gemseo.api import configure_logger, create_discipline, create_scenario
from gemseo.problems.sobieski.core import SobieskiProblem

IS_NT = os_name == "nt"

configure_logger()

##############################################################################
# Instantiate the  disciplines
# ----------------------------
# First, we instantiate the four disciplines of the use case:
# :class:`~gemseo.problems.sobieski.wrappers.SobieskiPropulsion`,
# :class:`~gemseo.problems.sobieski.wrappers.SobieskiAerodynamics`,
# :class:`~gemseo.problems.sobieski.wrappers.SobieskiMission`
# and :class:`~gemseo.problems.sobieski.wrappers.SobieskiStructure`.
disciplines = create_discipline(
    [
        "SobieskiPropulsion",
        "SobieskiAerodynamics",
        "SobieskiMission",
        "SobieskiStructure",
    ]
)

##############################################################################
# Build, execute and post-process the scenario
# --------------------------------------------
# Then, we build the scenario which links the disciplines
# with the formulation and the optimization algorithm. Here, we use the
# :class:`.BiLevel` formulation. We tell the scenario to minimize -y_4
# instead of minimizing y_4 (range), which is the default option.
#
# We need to define the design space.
design_space = SobieskiProblem().read_design_space()

##############################################################################
# Instantiate the scenario
# ^^^^^^^^^^^^^^^^^^^^^^^^
scenario = create_scenario(
    disciplines,
    formulation="MDF",
    objective_name="y_4",
    design_space=design_space,
    maximize_objective=True,
    scenario_type="DOE",
)
##############################################################################
# Set the design constraints
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
for constraint in ["g_1", "g_2", "g_3"]:
    scenario.add_constraint(constraint, "ineq")

##############################################################################
# Execute the scenario
# ^^^^^^^^^^^^^^^^^^^^
# Use provided analytic derivatives
scenario.set_differentiation_method("user")

n_processes = 4
if IS_NT:  # Under windows, don't do multiprocessing
    n_processes = 1

##############################################################################
# We define the algorithm options. Here the criterion = center option of pyDOE
# centers the points within the sampling intervals.
algo_options = {
    "criterion": "center",
    # Evaluate gradient of the MDA
    # with coupled adjoint
    "eval_jac": True,
    # Run in parallel on 4 processors
    "n_processes": n_processes,
}
run_inputs = {"n_samples": 30, "algo": "lhs", "algo_options": algo_options}
scenario.execute(run_inputs)

##############################################################################
# Plot the optimization history view
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
scenario.post_process("OptHistoryView", show=False, save=False)

##############################################################################
# Plot the scatter matrix
# ^^^^^^^^^^^^^^^^^^^^^^^
scenario.post_process(
    "ScatterPlotMatrix", show=False, save=False, variables_list=["y_4", "x_shared"]
)

##############################################################################
# Plot correlations
# ^^^^^^^^^^^^^^^^^
scenario.post_process("Correlations", show=False, save=False)
# Workaround for HTML rendering, instead of ``show=True``
plt.show()
