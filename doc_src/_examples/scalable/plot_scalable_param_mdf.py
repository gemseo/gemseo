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
Parametric scalable MDO problem - MDF
=====================================
We define a scalable problem based on two strongly coupled disciplines
and a weakly one, with the following properties:

- 3 shared design parameters,
- 2 local design parameters for the first strongly coupled discipline,
- 2 coupling variables for the first strongly coupled discipline,
- 4 local design parameters for the second strongly coupled discipline,
- 3 coupling variables for the second strongly coupled discipline.

We would like to solve this MDO problem by means of an MDF formulation.
"""
from __future__ import annotations

from gemseo.api import configure_logger
from gemseo.api import create_scenario
from gemseo.api import generate_n2_plot
from gemseo.problems.scalable.parametric.problem import TMScalableProblem

configure_logger()

#######################################################################################
# Instantiation of the scalable problem
# -------------------------------------
n_shared = 3
n_local = [2, 4]
n_coupling = [2, 3]
problem = TMScalableProblem(n_shared, n_local, n_coupling)

#######################################################################################
# Display the coupling structure
# ------------------------------
generate_n2_plot(problem.disciplines, save=False, show=True)

#######################################################################################
# Solve the MDO using an MDF formulation
# --------------------------------------
scenario = create_scenario(problem.disciplines, "MDF", "obj", problem.design_space)
scenario.add_constraint("cstr_0", "ineq")
scenario.add_constraint("cstr_1", "ineq")
scenario.execute({"algo": "NLOPT_SLSQP", "max_iter": 100})

#######################################################################################
# Post-process the results
# ------------------------
scenario.post_process("OptHistoryView", save=False, show=True)
