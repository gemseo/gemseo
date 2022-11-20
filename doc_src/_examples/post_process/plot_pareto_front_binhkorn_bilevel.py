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
Pareto front on Binh and Korn problem using a BiLevel formulation
=================================================================

In this example,
we illustrate the computation of a Pareto front plot for the Binh and Korn problem.
We use a BiLevel formulation in order to only compute the Pareto-optimal points.
"""
###############################################################################
# Import
# ------
# The first step is to import some functions from the API,
# and to configure the logger.
from __future__ import annotations

from gemseo.api import configure_logger
from gemseo.api import create_design_space
from gemseo.api import create_discipline
from gemseo.api import create_scenario
from gemseo.disciplines.scenario_adapter import MDOScenarioAdapter
from numpy import array

configure_logger()


###############################################################################
# Definition of the disciplines
# -----------------------------
#
# In this example,
# we create the Binh and Korn disciplines from scratch by declaring their expressions and using
# the :class:`.AnalyticDiscipline`.

expr_binh_korn = {
    "obj1": "4*x1**2 + 4*x2**2",
    "obj2": "(x1-5.)**2 + (x2-5.)**2",
    "cstr1": "(x1-5.)**2 + x2**2 - 25.",
    "cstr2": "-(x1-8.)**2 - (x2+3)**2 + 7.7",
}

################################################################################
# This constraint will be used to set `obj1` to a target value for the lower-level scenario.

expr_cstr_obj1_target = {"cstr3": "obj1 - obj1_target"}

###############################################################################
# Instantiation of the disciplines
# --------------------------------
#
# Here, we create the disciplines from their expressions.

discipline_binh_korn = create_discipline(
    "AnalyticDiscipline", expressions=expr_binh_korn
)
discipline_cstr_obj1 = create_discipline(
    "AnalyticDiscipline", expressions=expr_cstr_obj1_target
)

###############################################################################
# Definition of the lower-level design space
# ------------------------------------------

design_space = create_design_space()
design_space.add_variable("x1", l_b=array([0.0]), u_b=array([5.0]), value=array([2.0]))
design_space.add_variable("x2", l_b=array([-5.0]), u_b=array([3.0]), value=array([2.0]))

disciplines = [
    discipline_binh_korn,
    discipline_cstr_obj1,
]

###############################################################################
# Creation of the lower-level scenario
# ------------------------------------
# This scenario aims at finding the `obj2` optimal value for a specific value of `obj1`.

scenario = create_scenario(
    disciplines,
    "DisciplinaryOpt",
    design_space=design_space,
    objective_name="obj2",
)

scenario.default_inputs = {"algo": "NLOPT_SLSQP", "max_iter": 100}

###############################################################################
# We add the Binh and Korn problem constraints.
scenario.add_constraint("cstr1", "ineq")
scenario.add_constraint("cstr2", "ineq")

###############################################################################
# We add a constraint to force the value of `obj1` to `obj1_target`.
scenario.add_constraint("cstr3", "eq")

###############################################################################
# Creation of an MDOScenarioAdapter
# ---------------------------------
#
# An :class:`.MDOScenarioAdapter` is created to use the lower-level scenario as a discipline.
# This newly created discipline takes as input a target `obj_1`,
# and returns `obj1`, `obj2` and `cstr3`.
# The latter variable is used by the upper level scenario
# to check if `obj1` = `obj1_target` at the end of the lower-lever scenario execution.

scenario_adapter = MDOScenarioAdapter(
    scenario, ["obj1_target"], ["obj1", "obj2", "cstr3"]
)
design_space_doe = create_design_space()
design_space_doe.add_variable(
    "obj1_target", l_b=array([0.1]), u_b=array([100.0]), value=array([1.0])
)

###############################################################################
# Creation of a DOEScenario
# -------------------------
# Create a DOE Scenario, which will take as input the scenario adapter.
# It will perform a DOE over the `obj1_target` variable.
# Note that `obj2` shall be added as an observable of `scenario_doe`,
# otherwise it cannot be used by the ParetoFront post-processing.

scenario_doe = create_scenario(
    scenario_adapter,
    formulation="DisciplinaryOpt",
    objective_name="obj1",
    design_space=design_space_doe,
    scenario_type="DOE",
)
scenario_doe.add_constraint("cstr3", "eq")
scenario_doe.add_observable("obj2")

###############################################################################
# Run the scenario
# ----------------
# Finally, we run a full-factorial DOE using 100 samples and we run the post-processing.
run_inputs = {"n_samples": 50, "algo": "fullfact"}
scenario_doe.execute(run_inputs)
scenario_doe.post_process(
    "ParetoFront", objectives=["obj1", "obj2"], save=False, show=True
)
