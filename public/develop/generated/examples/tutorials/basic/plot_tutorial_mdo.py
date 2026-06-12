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
"""# Tutorial - Execute your first Multi-Disciplinary Optimization

## Goal

In this tutorial, you will create your first scenario
to make a multi-disciplary optimization.
You will focus on the Sobieski test case.

In GEMSEO, the [MDOScenario][gemseo.scenarios.mdo.MDOScenario]
is a central component dedicated to optimization problems.
Unlike the [EvaluationScenario][gemseo.scenarios.evaluation.EvaluationScenario],
it is specifically designed to define objective functions (to be minimized or maximized)
as well as constraints.
Here, you will create and exploit the
[MDOScenario][gemseo.scenarios.mdo.MDOScenario].
"""

from __future__ import annotations

from gemseo import create_discipline
from gemseo import generate_n2_plot
from gemseo.problems.mdo.sobieski.core.design_space import SobieskiDesignSpace
from gemseo.scenarios.mdo import MDOScenario
from gemseo.settings.formulations import MDF_Settings
from gemseo.settings.linear_solvers import LGMRES_Settings
from gemseo.settings.mda import MDAGaussSeidel_Settings
from gemseo.settings.opt import SLSQP_Settings
from gemseo.settings.post import OptHistoryView_Settings

# %%
# ## Step 1 — The disciplines
#
# The Sobieski test case is implemented in GEMSEO
# so that the disciplines can be fetched.
# Here, you can simply create the disciplines using their names:
disciplines = create_discipline([
    "SobieskiPropulsion",
    "SobieskiAerodynamics",
    "SobieskiMission",
    "SobieskiStructure",
])
# %%
# You can quickly access the most relevant information of any discipline (name, inputs,
# and outputs) with Python's `print()` function.
# Moreover,
# you can get the default input values of a discipline
# with the attribute
# [Discipline.default_input_data][gemseo.core.discipline.discipline.Discipline.default_input_data]
for discipline in disciplines:
    print(discipline)  # noqa: T201
    print(f"Default inputs: {discipline.default_input_data}")  # noqa: T201
    print()  # noqa: T201

# %%
# You may also be interested in plotting the couplings of your disciplines.
#
# !!! note
#     A much more detailed explanation of coupling
#     visualization is available [here][concept-coupling-visualization].
generate_n2_plot(disciplines, save=False, show=True)

# %%
# ## Step 2 - Create the scenario
#
# The design space of the Sobieski test case is already implemented in GEMSEO.
# You first create and visualize it.
design_space = SobieskiDesignSpace()
design_space

# %%
# Then,
# you build the [MDOScenario][gemseo.scenarios.mdo.MDOScenario]
# which links the disciplines together according to the chosen formulation.
# Here,
# you use the
# [MDF][gemseo.formulations.mdf.MDF] formulation.
#
# !!! tip
#     The [MDF][gemseo.formulations.mdf.MDF] formulation is often recommanded
#     when you start your MDO process.
#     More complex formulations can be tried in a second phase,
#     if the [MDF][gemseo.formulations.mdf.MDF] formulation
#     does not help to solve the problem.
#
# !!! note
#     The list of the available formulations, and their description, is available in
#     [MDO formulations][concept-mdo-formulations].
#
# The [MDF][gemseo.formulations.mdf.MDF] formulation will create MDAs in the workflow,
# if needed.
# You can parametrize the MDAs thanks to the settings.
scenario = MDOScenario(
    disciplines,
    design_space,
    formulation_settings=MDF_Settings(
        main_mda_settings=MDAGaussSeidel_Settings(
            tolerance=1e-14,
            max_mda_iter=50,
            warm_start=True,
            linear_solver_settings=LGMRES_Settings(rtol=1e-14),
        )
    ),
)

# %%
# Once your [MDOScenario][gemseo.scenarios.mdo.MDOScenario] has been created,
# you must define the objective functions with the
# [add_objective()][gemseo.scenarios.mdo.MDOScenario.add_objective] method.
scenario.add_objective("y_4", minimize=False)

# %%
# !!! note
#     You could have chosen the [IDF][gemseo.formulations.idf.IDF] formulation instead,
#     by using the [IDF_Settings][gemseo.formulations.idf_settings.IDF_Settings].
#     This would have automatically generated another workflow,
#     allowing the use of multi-processing IDF features.
#
# You can add different optimization constraints in your scenario.
# For instance,
# if you intend to use the Sobieski constraints,
# e.g. $g_1 \leq 0$, $g_2 \leq 0$, and $g_3 \leq 0$,
# you must use the
# [add_constraint()][gemseo.scenarios.mdo.MDOScenario.add_constraint] method.
for c_name in ["g_1", "g_2", "g_3"]:
    scenario.add_constraint(c_name, constraint_type="ineq")

# %%
# Once your scenario looks fine,
# you may want to generate the XDSM
# to visualize your workflow.
scenario.xdsmize(save_html=False)

# %%
# ## Step 3 - Execute your scenario
#
# Here, you can use the `SLSQP` algorithm to find the solution of your problem.
# You limit the number of iteration to 10, and you modify some convergence tolerances.
# Finally, you want your design space to be normalized while using this algorithm,
# this is recommended and activated by default.
scenario.execute(
    SLSQP_Settings(
        max_iter=10,
        ftol_rel=1e-10,
        ineq_tolerance=2e-3,
        normalize_design_space=True,
    )
)

# %%
# !!! note
#     There are different ways to solve an [MDOScenario][gemseo.scenarios.mdo.MDOScenario].
#     You can either chose an optimization algorithm, like `SLSQP`,
#     or you can chose a Design of Experiment algorithm.
#     In that case,
#     the functions of interest will be evaluated in different points
#     according to a sampling strategy,
#     and the best point will be considered as optimal.
#
# !!! how-to
#     There are different ways to save your optimization process.
#     You can either save during the process,
#     to reduce the risk of something going wrong.
#     or you can save at the end of the optimization.
#     For details,
#     see [Save the execution history of a scenario][save-the-execution-history-of-a-scenario].
#
# ## Step 4 - Post-process your result
#
# The optimum results can be found in the execution log. It is also possible to
# access them with
# [MDOScenario.optimization_result][gemseo.scenarios.mdo.MDOScenario.optimization_result].
scenario.optimization_result

# %%
# The scenario can also be post-processed with the
# [post_process()][gemseo.scenarios.mdo.MDOScenario.post_process]] method.
scenario.post_process(OptHistoryView_Settings(save=False, show=True))

# %%
# ## Key takeaways
#
# You've learnt to combine different important concepts of GEMSEO
# to create your first optimisation workflow.
# The [MDOScenario][gemseo.scenarios.mdo.MDOScenario] uses a
# [MDO formulation][concept-mdo-formulations] which is in charge of building the process
# with the [Disciplines][gemseo.core.discipline.discipline.Discipline],
# where the way how couplings are solved depends on the selected strategy.
#
# When executed with a given algorithm,
# your scenario can be post-processed to visualize your results.
#
# ## How-to guides
#
# For further information,
# please refer to the following how-to guides:
#
# - [Generate an XDSM chart][generate-an-xdsm-chart] ,
# - [Save the execution history of a scenario][save-the-execution-history-of-a-scenario],
# - [Post-process an optimization scenario][post-process-a-scenario].
