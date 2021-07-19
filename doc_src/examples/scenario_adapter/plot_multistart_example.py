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
#        :author: Arthur Piat, Francois Gallard
#
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Multistart optimization
=======================

Runs simple optimization problem with multiple starting points
Nests a :class:`.MDOScenario` in a :class:`.DOEScenario`
using a :class:`.MDOScenarioAdapter`.
"""
from __future__ import division, unicode_literals

from matplotlib import pyplot as plt

from gemseo.api import (
    configure_logger,
    create_design_space,
    create_discipline,
    create_scenario,
)
from gemseo.core.mdo_scenario import MDOScenarioAdapter

configure_logger()


##############################################################################
# Create the disciplines
# ----------------------
objective = create_discipline(
    "AnalyticDiscipline", expressions_dict={"obj": "x**3-x+1"}
)
constraint = create_discipline(
    "AnalyticDiscipline", expressions_dict={"cstr": "x**2+obj**2-1.5"}
)

##############################################################################
# Create the design space
# -----------------------
design_space = create_design_space()
design_space.add_variable("x", 1, l_b=-1.5, u_b=1.5, value=1.5)

##############################################################################
# Create the MDO scenario
# -----------------------
scenario = create_scenario(
    [objective, constraint],
    formulation="DisciplinaryOpt",
    objective_name="obj",
    design_space=design_space,
)
scenario.default_inputs = {"algo": "SLSQP", "max_iter": 10}
scenario.add_constraint("cstr", "ineq")

##############################################################################
# Create the scenario adapter
# ---------------------------
dv_names = scenario.formulation.opt_problem.design_space.variables_names
adapter = MDOScenarioAdapter(
    scenario, dv_names, ["obj", "cstr"], set_x0_before_opt=True
)

##############################################################################
# Create the DOE scenario
# -----------------------
scenario_doe = create_scenario(
    adapter,
    formulation="DisciplinaryOpt",
    objective_name="obj",
    design_space=design_space,
    scenario_type="DOE",
)
scenario_doe.add_constraint("cstr", "ineq")
run_inputs = {"n_samples": 10, "algo": "fullfact"}
scenario_doe.execute(run_inputs)

##############################################################################
# Plot the optimum objective for different x0
# -------------------------------------------
scenario_doe.post_process("BasicHistory", data_list=["obj"], save=False, show=False)
# Workaround for HTML rendering, instead of ``show=True``
plt.show()
