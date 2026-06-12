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
"""# Tutorial - Solve a bi-level MDO problem

## Goal

In this tutorial,
you will learn how to solve a bi-level MDO problem.
You will focus on the Sobieski test case.

!!! note
    This advanced tutorial is based on the same test case studied in
    [Tutorial - Execute your first Multi-Disciplinary Optimization][tutorial-execute-your-first-multi-disciplinary-optimization].

"""

from __future__ import annotations

from logging import WARNING

from gemseo import create_discipline
from gemseo.problems.mdo.sobieski.core.design_space import SobieskiDesignSpace
from gemseo.scenarios.mdo import MDOScenario
from gemseo.settings.formulations import BiLevel_Settings
from gemseo.settings.linear_solvers import LGMRES_Settings
from gemseo.settings.mda import MDAGaussSeidel_Settings
from gemseo.settings.opt import NLOPT_COBYLA_Settings
from gemseo.settings.opt import SLSQP_Settings
from gemseo.settings.post import OptHistoryView_Settings

# %%
# ## Step 1 - Instantiate the  disciplines and the design space
#
# First, we instantiate the four disciplines of the use case:
# [SobieskiPropulsion][gemseo.problems.mdo.sobieski.disciplines.SobieskiPropulsion],
# [SobieskiAerodynamics][gemseo.problems.mdo.sobieski.disciplines.SobieskiAerodynamics],
# [SobieskiMission][gemseo.problems.mdo.sobieski.disciplines.SobieskiMission]
# and [SobieskiStructure][gemseo.problems.mdo.sobieski.disciplines.SobieskiStructure].
propulsion, aerodynamics, mission, structure = create_discipline([
    "SobieskiPropulsion",
    "SobieskiAerodynamics",
    "SobieskiMission",
    "SobieskiStructure",
])

# %%
# Then, we build the design space.
design_space = SobieskiDesignSpace()
design_space

# %%
# Step 2 - Create the sub-scenarios
#
# Then, we build a sub-scenario for each disciplinary optimization,
# using the following algorithm, maximum number of iterations and
# algorithm settings:

slsqp_settings = SLSQP_Settings(
    max_iter=30,
    xtol_rel=1e-7,
    xtol_abs=1e-7,
    ftol_rel=1e-7,
    ftol_abs=1e-7,
    ineq_tolerance=1e-4,
)


# %%
# ### Build a sub-scenario for Propulsion
# This sub-scenario will minimize SFC.
sc_prop = MDOScenario(
    (propulsion,), design_space.filter("x_3", copy=True), name="PropulsionScenario"
)
sc_prop.add_objective("y_34")
sc_prop.set_algorithm(slsqp_settings)
sc_prop.add_constraint("g_3", constraint_type="ineq")

# %%
# ### Build a sub-scenario for Aerodynamics
# This sub-scenario will minimize L/D.
sc_aero = MDOScenario(
    (aerodynamics,), design_space.filter("x_2", copy=True), name="AerodynamicsScenario"
)
sc_aero.add_objective("y_24", minimize=False)
sc_aero.set_algorithm(slsqp_settings)
sc_aero.add_constraint("g_2", constraint_type="ineq")

# %%
# ### Build a sub-scenario for Structure
# This sub-scenario will maximize
# log(aircraft total weight / (aircraft total weight - fuel weight)).
sc_str = MDOScenario(
    (structure,),
    design_space.filter("x_1", copy=True),
    name="StructureScenario",
)
sc_str.add_objective("y_11", minimize=False)
sc_str.add_constraint("g_1", constraint_type="ineq")
sc_str.set_algorithm(slsqp_settings)

# %%
# ## Step 3 - The BiLevel scenario
# This scenario is based on the three previous sub-scenarios and on the
# Mission and aims to maximize the range (Breguet).
#
# The `disciplines` argument of the
# [create_scenario()][gemseo.create_scenario] function gathers both
# [disciplines][gemseo.core.discipline.discipline.Discipline] and
# [MDOScenario][gemseo.scenarios.mdo.MDOScenario].
# The [BiLevel formulation][the-bi-level-formulation] will automatically transform the scenarios into disciplines.
#
# !!! note
#     An [MDOScenario][gemseo.scenarios.mdo.MDOScenario]
#     can be transformed into a
#     [Discipline][gemseo.core.discipline.discipline.Discipline] thanks to the
#     [ScenarioAdapter][gemseo.disciplines.scenario_adapters.mdo_scenario_adapter.MDOScenarioAdapter].

system_scenario = MDOScenario(
    (sc_prop, sc_aero, sc_str, mission),
    design_space.filter("x_shared", copy=True),
    formulation_settings=BiLevel_Settings(
        apply_constraints_to_sub_scenarios=False,
        parallel_scenarios=False,
        multithread_scenarios=True,
        main_mda_settings=MDAGaussSeidel_Settings(
            tolerance=1e-14,
            max_mda_iter=50,
            warm_start=True,
            linear_solver_settings=LGMRES_Settings(rtol=1e-14),
        ),
        sub_scenarios_log_level=WARNING,
    ),
)
system_scenario.add_objective("y_4", minimize=False)
system_scenario.add_constraint(["g_1", "g_2", "g_3"], constraint_type="ineq")

# %%
# !!! tip
#     When running BiLevel scenarios, it is interesting to access the optimization
#     history of the sub-scenarios for each system iteration. By default, the `keep_opt_history` setting
#     is set to `True` which allows you to store in memory the
#     databases of the sub-scenarios (see the last section of this example for more
#     details).
#     Sometimes, storing the databases in memory uses up too much space which can cause
#     performance issues. In such cases, set `keep_opt_history=False` and save the
#     databases to the disk, through hdf files,  using `save_opt_history=True`.

# %%
# The XDSM can be generated to visualize the workflow:
system_scenario.xdsmize(save_html=False)

# %%
# Finally the scenario can be executed with a gradient-free algorithm
# since the inner problem with sub-scenarios cannot be differentiated.
# Here,
# the `system_scenario` is limited to 140 iterations.
# When a sub-scenario is executed,
# GEMSEO understands that a specific optimization must be performed,
# which means that
# at each iteration of the `system_scenario`,
# GEMSEO executes 3 different optimization scenarios:
# - Propulsion
# - Aerodynamics
# - Structure
system_scenario.execute(
    NLOPT_COBYLA_Settings(
        max_iter=140,
        xtol_rel=1e-7,
        xtol_abs=1e-7,
        ftol_rel=1e-7,
        ftol_abs=1e-7,
        ineq_tolerance=1e-4,
    )
)

# %%
# ## Step 4 - The system scenario results
#
# From the system-level point of view,
# the results can be retrieved with the conventionnal way:
system_scenario.optimization_result

# %%
# Post-processes can also be applied to your system scenario.
system_scenario.post_process(OptHistoryView_Settings(save=False, show=True))

# %%
# ## Step 5 - The results of inner optimizations
#
# Different optimization sub-scenarios were wrapped into disciplines
# through a scenario adapter.
# Theses sub-scenarios can be retrieved from
# [system_scenario.formulation.scenario_adapters][gemseo.formulations.bilevel.BiLevel.scenario_adapters] property.
# Then the results of sub-scenarios can be obtained and any post-processings can be applied.
for sub_scenario in system_scenario.formulation.scenario_adapters:
    print(sub_scenario)

# %%
# For instance, the optimization histories (at each iteration of the system level)
# of the `structure` scenario can be retrieved,
# so the number of databases is given by the number of system-level iterations.
structure_databases = system_scenario.formulation.scenario_adapters[2].databases
len(structure_databases)

# %%
# !!! note
#     Post-processes can be applied to the database created at the optimal point.
#     To do so,
#     the index must be found using the
#     `system_scenario.optimization_result.optimum_index` index.
#
# More structured results can be retrieved from the
# [get_result()][gemseo.scenarios.mdo.MDOScenario.get_result] method,
# which gives an easier access to the system and sub-level results.
# In the BiLevel case, this method will return a
# [BiLevelScenarioResult][gemseo.scenarios.scenario_results.bilevel_scenario_result.BiLevelScenarioResult].
#
# !!! note
#     The value contained in `system_scenario.optimization_result` can be retrieved by this mean as well.

bilevel_result = system_scenario.get_result()
bilevel_result.get_top_optimization_result()

# %%
# Let's focus on the Structure optimization.
# The structure optimization can be found at index=2.

structure_result = bilevel_result.get_sub_optimization_result(2)
structure_result

# %%
#
# ## Key takeaways
#
# You've created a BiLevel-based workflow
# for solving a Multi-Disciplinary Optimization problem.
# That way,
# you are able to design a process that embeds
# several sub-optimization processes,
# which are run through disciplines.

#
# ## How-to guides
#
# For further information,
# please refer to the following how-to guides:
#
# - [Transform a scenario into a discipline][transform-a-scenario-into-a-discipline].
# - Post-process a database (TODO: create how-to)
