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
from __future__ import division, unicode_literals

from gemseo.api import configure_logger, generate_n2_plot
from gemseo.problems.scalable.parametric.problem import TMScalableProblem
from gemseo.uncertainty.umdo.umdo_scenario import UMDOScenario

configure_logger()

#######################################################################################
# Instantiation of the scalable problem
# -------------------------------------
n_shared = 3
n_local = [2, 4]
n_coupling = [2, 3]
problem = TMScalableProblem(n_shared, n_local, n_coupling, noised_coupling=True)

#######################################################################################
# Display the coupling structure
# ------------------------------
generate_n2_plot(problem.disciplines, save=False, show=True)

#######################################################################################
# Solve the U-MDO using an MDF formulation
# ----------------------------------------
sampling_options = {"algo": "OT_MONTE_CARLO", "n_samples": 100}
scenario = UMDOScenario(
    problem.disciplines,
    "UMDF",
    "obj",
    problem.design_space,
    sampling_options=sampling_options,
)
#######################################################################################
# We set the robustness measures for the objective and the constraints.
scenario.set_robustness_measure("obj", "mean")
scenario.set_robustness_measure("cstr_0", "mean_std", std_factor=3.0)
scenario.set_robustness_measure("cstr_1", "mean_std", std_factor=3.0)
scenario.add_constraint("cstr_0", "ineq", "p_cstr_0", value=0.0)
scenario.add_constraint("cstr_1", "ineq", "p_cstr_1", value=0.0)

#######################################################################################
# Display XDSM
scenario.xdsmize(latex_output=True)

#######################################################################################
# Execute
scenario.execute({"algo": "NLOPT_COBYLA", "max_iter": 100, "xtol_abs": 1e-3})

#######################################################################################
# Post-process the results
# ------------------------
scenario.post_process("OptHistoryView", save=False, show=True)
