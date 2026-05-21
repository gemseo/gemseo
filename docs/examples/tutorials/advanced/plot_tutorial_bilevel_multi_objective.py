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
#    INITIAL AUTHORS - initial API and implementation and/or initial documentation
#        :author: Jean-Christophe Giret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""# Tutorial - Compute a Pareto front with a BiLevel formulation

## Goal

This tutorial shows an alternative approach to computing a Pareto front
for a multi-objective problem — without a dedicated multi-objective algorithm.

!!! how-to
    You can also use algorithms supporting multi-objective problems.
    See [Use the mNBI algorithm][] for details.

!!! note
    If you want to solve multi-objective problems,
    you can also have a look into
    the [gemseo-pymoo](https://gitlab.com/gemseo/dev/gemseo-pymoo) plugin.

The idea is to treat the first objective $f_1$ as a system-level DOE parameter
(a target value `obj1_target`) and minimize the second objective $f_2$ at the
sub-level for each target value.
Only sub-level solutions that match the target appear on the Pareto front.

We use the [BiLevel][gemseo.formulations.bilevel.BiLevel] formulation
to nest a lower-level [MDOScenario][gemseo.scenarios.mdo.MDOScenario]
(minimizing $f_2$ for a given `obj1_target`)
inside a system-level [MDOScenario][gemseo.scenarios.mdo.MDOScenario]
solved by a DoE
(sweeping `obj1_target`).

The test case is the Binh-Korn problem:

$$
   \\begin{aligned}
   \\text{minimize} & \\; f_1(x_1, x_2) = 4x_1^2 + 4x_2^2 \\\\
                   & \\; f_2(x_1, x_2) = (x_1-5)^2 + (x_2-5)^2 \\\\
   \\text{subject to} & \\; g_1 = (x_1-5)^2 + x_2^2 - 25 \\leq 0 \\\\
                     & \\; g_2 = -(x_1-8)^2 - (x_2+3)^2 + 7.7 \\leq 0 \\\\
                     & \\; 0 \\leq x_1 \\leq 5,\\quad 0 \\leq x_2 \\leq 3
   \\end{aligned}
$$
"""

from __future__ import annotations

from copy import deepcopy
from logging import WARNING

from numpy import array

from gemseo import execute_post
from gemseo.algos.design_space import DesignSpace
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.scenarios.mdo import MDOScenario
from gemseo.settings.doe import PYDOE_FULLFACT_Settings
from gemseo.settings.formulations import BiLevel_Settings
from gemseo.settings.opt import NLOPT_SLSQP_Settings
from gemseo.settings.post import OptHistoryView_Settings
from gemseo.settings.post import ParetoFront_Settings

# %%
# ## Step 1 - Define the disciplines
#
# We define two
# [AnalyticDiscipline][gemseo.disciplines.analytic.AnalyticDiscipline] objects
# from symbolic expressions.
# The first discipline computes $f_1$, $f_2$, $g_1$, and $g_2$.
expr_binh_korn = {
    "obj1": "4*x1**2 + 4*x2**2",
    "obj2": "(x1-5.)**2 + (x2-5.)**2",
    "cstr1": "(x1-5.)**2 + x2**2 - 25.",
    "cstr2": "-(x1-8.)**2 - (x2+3)**2 + 7.7",
}
discipline_binh_korn = AnalyticDiscipline(expr_binh_korn, name="Binh Korn")

# %%
# The second discipline computes the gap between $f_1$ and its target value.
# This gap is enforced as a constraint at the sub-level to pin $f_1$
# to the value chosen by the system-level DOE.
expr_cstr_obj1_target = {"cstr3": "obj1 - obj1_target"}
discipline_cstr_obj1 = AnalyticDiscipline(expr_cstr_obj1_target, name="Constr")

# %%
# ## Step 2 - Create the lower-level design space
#
# The lower-level design space contains the physical design variables
# $x_1$ and $x_2$.
design_space = DesignSpace()
design_space.add_variable(
    "x1", lower_bound=array([0.0]), upper_bound=array([5.0]), value=array([2.0])
)
design_space.add_variable(
    "x2", lower_bound=array([-5.0]), upper_bound=array([3.0]), value=array([2.0])
)

# %%
# ## Step 3 - Create the lower-level scenario
#
# This scenario minimizes $f_2$ for a fixed value of `obj1_target`.
# The `DisciplinaryOpt` formulation is used with NLOPT SLSQP as the solver.
disciplines = [discipline_binh_korn, discipline_cstr_obj1]
sub_scenario = MDOScenario(
    disciplines,
    design_space,
)
sub_scenario.add_objective("obj2")
sub_scenario.set_algorithm(NLOPT_SLSQP_Settings(max_iter=50))

# %%
# We add the Binh-Korn inequality constraints $g_1$ and $g_2$.
sub_scenario.add_constraint("cstr1", constraint_type="ineq")
sub_scenario.add_constraint("cstr2", constraint_type="ineq")

# %%
# ## Step 4 - Create the system-level design space
#
# At the system level, the only variable is `obj1_target`:
# the value of $f_1$ swept by the DOE.
system_design_space = DesignSpace()
system_design_space.add_variable(
    "obj1_target",
    lower_bound=array([0.1]),
    upper_bound=array([100.0]),
    value=array([1.0]),
)

# %%
# ## Step 5 - Create the system-level DOE scenario
#
# The [BiLevel][gemseo.formulations.bilevel.BiLevel] formulation
# wraps the lower-level scenario inside the system-level DOE.
# For each value of `obj1_target` sampled by the DOE,
# GEMSEO runs the lower-level optimization and records the result.

# %%
# !!! tip
#     By default, `keep_opt_history=True` stores the sub-scenario databases in memory,
#     allowing you to inspect each sub-level optimization history (see Step 8).
#     If memory is a concern, set `keep_opt_history=False` and use
#     `save_opt_history=True` to write databases to disk instead.
#     When running sub-scenarios in parallel
#     (`n_processes > 1`),
#     set `naming="UUID"` for multiprocessing-safe file names.
#     In parallel mode, `keep_opt_history` does not work
#     because databases cannot be copied across processes.
system_scenario = MDOScenario(
    [sub_scenario],
    system_design_space,
    formulation_settings=BiLevel_Settings(sub_scenarios_log_level=WARNING),
)
system_scenario.add_objective("obj1")

# %%
# ## Step 6 - Add the system-level constraint and observable
#
# The constraint `cstr3` enforces that the sub-level solution actually achieves
# `obj1_target`.
# The [BiLevel][gemseo.formulations.bilevel.BiLevel] formulation automatically
# propagates this system-level constraint to the lower-level scenario.
# Adding `obj2` as an observable makes it available for post-processing.
system_scenario.add_constraint("cstr3")
system_scenario.add_observable("obj2")

# %%
# The xDSM diagram shows the nested structure: the DOE drives the sub-scenario.
system_scenario.xdsmize(show_html=True, save_html=False)

# %%
# ## Step 7 - Run the scenario
#
# A full-factorial DOE with 20 samples sweeps the range of `obj1_target`.
# Each sample triggers one lower-level optimization.
system_scenario.execute(PYDOE_FULLFACT_Settings(n_samples=20))

# %%
# The Pareto front is assembled from the feasible DOE results:
# each point is a $(f_1, f_2)$ pair where $f_1 \approx \texttt{obj1\_target}$.
system_scenario.post_process(
    ParetoFront_Settings(objectives=["obj1", "obj2"], save=False, show=True)
)

# %%
# ## Step 8 - Inspect the sub-scenario optimization histories
#
# Since `keep_opt_history=True` (default), the database of each sub-level run
# is stored in memory. We inspect the first two.

# %%
# !!! note
#
#     This section does not work if `system_scenario` was run with `n_processes > 1`.
#     In that case, retrieve the databases from disk (see the tip in Step 5).
sub_scenario_databases = system_scenario.formulation.scenario_adapters[0].databases
for database in sub_scenario_databases[:2]:
    opt_problem = deepcopy(sub_scenario.formulation.problem)
    opt_problem.database = database
    execute_post(opt_problem, OptHistoryView_Settings(save=False, show=True))

# %%
# ## Key takeaways
#
# You learnt to compute a Pareto front with the
# [BiLevel][gemseo.formulations.bilevel.BiLevel] formulation,
# without a dedicated multi-objective algorithm.
# The core idea is to fix $f_1$ as a DOE parameter at the system level
# and minimize $f_2$ at the sub-level for each target value.
#
# This approach gives full control over both the system-level sampling
# and the sub-level solver, but requires two nested scenarios.
#
# ## How-to guides
#
# For further information,
# please refer to the following how-to guides:
#
# - [Use the mNBI algorithm for solving multi-objective problems][use-the-mnbi-algorithm]
