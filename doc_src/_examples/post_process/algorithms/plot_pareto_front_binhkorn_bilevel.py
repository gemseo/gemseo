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
Pareto front on the Binh and Korn problem using a BiLevel formulation
=====================================================================

In this example,
we illustrate the computation of a Pareto front plot for the Binh and Korn problem.
We use a BiLevel formulation in order to only compute the Pareto-optimal points.
"""

# %%
# Import
# ------
# The first step is to import some high-level functions
# and to configure the logger.
from __future__ import annotations

from logging import WARNING

from numpy import array

from gemseo import configure_logger
from gemseo import create_design_space
from gemseo import create_discipline
from gemseo import create_scenario

configure_logger()


# %%
# Definition of the disciplines
# -----------------------------
#
# In this example,
# we create the Binh and Korn disciplines from scratch by declaring
# their expressions and using the :class:`.AnalyticDiscipline`.

expr_binh_korn = {
    "obj1": "4*x1**2 + 4*x2**2",
    "obj2": "(x1-5.)**2 + (x2-5.)**2",
    "cstr1": "(x1-5.)**2 + x2**2 - 25.",
    "cstr2": "-(x1-8.)**2 - (x2+3)**2 + 7.7",
}

# %%
# This constraint will be used to set `obj1` to a target value for
# the lower-level scenario.

expr_cstr_obj1_target = {"cstr3": "obj1 - obj1_target"}

# %%
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

# %%
# Definition of the lower-level design space
# ------------------------------------------

design_space = create_design_space()
design_space.add_variable("x1", l_b=array([0.0]), u_b=array([5.0]), value=array([2.0]))
design_space.add_variable("x2", l_b=array([-5.0]), u_b=array([3.0]), value=array([2.0]))

disciplines = [
    discipline_binh_korn,
    discipline_cstr_obj1,
]

# %%
# Creation of the lower-level scenario
# ------------------------------------
# This scenario aims at finding the `obj2` optimal value for a specific value of `obj1`.

sub_scenario = create_scenario(
    disciplines,
    "DisciplinaryOpt",
    design_space=design_space,
    objective_name="obj2",
)

sub_scenario.default_inputs = {"algo": "NLOPT_SLSQP", "max_iter": 100}

# %%
# We add the Binh and Korn problem constraints.
sub_scenario.add_constraint("cstr1", "ineq")
sub_scenario.add_constraint("cstr2", "ineq")

# %%
# Creation of the design space for the system-level scenario
# ----------------------------------------------------------
# At the system level, we will fix a target for the `obj1` value of
# the lower-level scenario.
system_design_space = create_design_space()
system_design_space.add_variable(
    "obj1_target", l_b=array([0.1]), u_b=array([100.0]), value=array([1.0])
)

# %%
# Creation of the system-level DOE Scenario
# -----------------------------------------
# The system-level scenario will perform a DOE over the `obj1_target` variable.
# We will use the `BiLevel` formulation to nest the lower-level scenario into the DOE.
# The log level for the sub scenarios is set to `WARNING` to avoid getting the complete
# log of each sub scenario, which would be too verbose. Set it to `INFO` if you wish to
# keep the logs of each sub scenario as well.

system_scenario = create_scenario(
    sub_scenario,
    formulation="BiLevel",
    objective_name="obj1",
    design_space=system_design_space,
    scenario_type="DOE",
    sub_scenarios_log_level=WARNING,
)

# %%
# Add the system-level constraint and observables
# -----------------------------------------------
# Here, we add the constraint on the `obj1_target`, this way we make sure that the
# lower-level scenario will respect the target imposed by the system.
# The BiLevel formulation will automatically add the constraints from the system-level
# to the lower-level, if you wish to handle the constraints manually, pass
# `apply_cstr_tosub_scenarios=False` as an argument to `create_scenario`.
# Note that `obj2` shall be added as an observable of `scenario_doe`,
# otherwise it cannot be used by the ParetoFront post-processing.
system_scenario.add_constraint("cstr3", "eq")
system_scenario.add_observable("obj2")
system_scenario.xdsmize()
# %%
# Run the scenario
# ----------------
# Finally, we run a full-factorial DOE using 100 samples and run the post-processing.
run_inputs = {"n_samples": 50, "algo": "fullfact"}
system_scenario.execute(run_inputs)
system_scenario.post_process(
    "ParetoFront", objectives=["obj1", "obj2"], save=False, show=True
)
