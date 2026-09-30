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
"""# Post-process the sub-scenario histories of a bi-level scenario

## Problem

With a [BiLevel][gemseo.formulation.bilevel.BiLevel] formulation,
a sub-scenario is executed at each system-level evaluation,
possibly several times, or not at all after a cache hit,
each time solving a full optimization problem.
You want to analyse the optimization histories of all these executions together,
for instance to compare how fast the sub-scenario converges
from one system-level evaluation to another.

## Solution

When `BiLevel_Settings.keep_opt_history` (enabled by default)
and/or `BiLevel_Settings.save_opt_history` are enabled,
the optimization history of a sub-scenario is retained after every execution.
Then,
[get_result()][gemseo.scenario.mdo.MDOScenario.get_result] returns a
[BiLevelScenarioResult][gemseo.scenario.scenario_result.bilevel_scenario_result.BiLevelScenarioResult]
whose
[get_sub_scenario_history_dataset()][gemseo.scenario.scenario_result.bilevel_scenario_result.BiLevelScenarioResult.get_sub_scenario_history_dataset]
combines the retained histories,
across all the executions of a sub-scenario,
into a single [Dataset][gemseo.dataset.dataset.Dataset]
— one row per sub-scenario iteration.

As this dataset stacks several independent optimization histories,
it is a plain [Dataset][gemseo.dataset.dataset.Dataset]
carrying no optimization metadata;
it is therefore not meant to be passed to [execute_post][gemseo.execute_post].

!!! warning
    The sub-scenario histories are not collected
    when the sub-scenarios are executed in separate processes,
    for example with `BiLevel_Settings(parallel_scenarios=True, multithread_scenarios=False)`,
    or with a system-level DOE that uses several processes.

## Step-by-step guide
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from numpy import array

from gemseo.discipline import AnalyticDiscipline
from gemseo.doe import CustomDOE_Settings
from gemseo.formulation import BiLevel_Settings
from gemseo.optimization import SLSQP_Settings
from gemseo.scenario import MDOScenario
from gemseo.space import DesignSpace

# %%
# ### Prerequisites
#
# This how-to needs a [BiLevel][gemseo.formulation.bilevel.BiLevel] scenario.
#
# The sub-scenario minimizes $z=(y-x)^2+x$ over its design variable $y$,
# for a value of $x$ passed down from the system level.
discipline = AnalyticDiscipline({"z": "(y - x) ** 2 + x"})
sub_design_space = DesignSpace()
sub_design_space.add_variable("y", lower_bound=-5.0, upper_bound=5.0, value=0.5)

sub_scenario = MDOScenario([discipline], sub_design_space)
sub_scenario.add_objective("z")
sub_scenario.set_algorithm(SLSQP_Settings(max_iter=10))

# %%
# The system level explores three values of $x$ with a
# [CustomDOE][gemseo.doe.custom_doe.custom_doe.CustomDOE],
# each of them triggering one execution of the sub-scenario.
system_design_space = DesignSpace()
system_design_space.add_variable("x", lower_bound=0.0, upper_bound=2.0)

system_scenario = MDOScenario(
    [sub_scenario],
    system_design_space,
    formulation_settings=BiLevel_Settings(),
)
system_scenario.add_objective("z")

# %%
# ### 1. Execute the scenario and get the sub-scenario history dataset
#
# [execute()][gemseo.scenario.mdo.MDOScenario.execute] the system-level scenario,
# then get the result and the history dataset of the sub-scenario,
# here the only one, hence index `0`.
system_scenario.execute(CustomDOE_Settings(samples=array([[0.0], [1.0], [2.0]])))
scenario_result = system_scenario.get_result()
history_dataset = scenario_result.get_sub_scenario_history_dataset(0)
history_dataset

# %%
# ### 2. Identify the column groups
#
# The columns of `history_dataset` are grouped as follows:
#
# | Group | Variables | Meaning |
# |---|---|---|
# | `designs` | the sub-scenario design variables | one column per component |
# | `objectives` | the sub-scenario objective(s) | from `OptimizationProblem.to_dataset` |
# | `equality_constraints`, `inequality_constraints` | the sub-scenario constraints, if any | from `OptimizationProblem.to_dataset` |
# | `observables` | the sub-scenario observables, if any | from `OptimizationProblem.to_dataset` |
# | `executions` (`BiLevelScenarioResult.executions_group`) | `execution`, `sub_iteration` | 1-based execution and iteration numbers |
# | `upper_level_designs` (`BiLevelScenarioResult.upper_level_designs_group`) | the upper-level variables | the values passed to the sub-scenario for that execution |

# %%
# ### 3. Extract subsets of the dataset
#
# Use [Dataset.get_view][gemseo.dataset.dataset.Dataset.get_view]
# to extract a subset of the columns,
# e.g. the design and objective columns.
designs_and_objectives = history_dataset.get_view(group_names=["designs", "objectives"])
designs_and_objectives

# %%
# Plain pandas indexing extracts a subset of the rows,
# e.g. those of a single execution.
first_execution = history_dataset[history_dataset["executions", "execution", 0] == 1]
first_execution

# %%
# ### 4. Compare the sub-scenario convergence across executions
#
# A simple way to compare the sub-scenario convergence across executions
# is to plot the objective against the sub-iteration, one line per execution,
# labelled with the upper-level value of $x$ that was used for that execution.
fig, ax = plt.subplots()
for _execution, group in history_dataset.groupby(("executions", "execution", 0)):
    ax.plot(
        group["executions", "sub_iteration", 0],
        group["objectives", "z", 0],
        marker="o",
        label=f"x = {group['upper_level_designs', 'x', 0].iloc[0]}",
    )
ax.set_xlabel("Sub-scenario iteration")
ax.set_ylabel("z")
ax.legend()
plt.show()

# %%
# ## Summary
#
# Enabling `BiLevel_Settings.keep_opt_history` (the default)
# and/or `BiLevel_Settings.save_opt_history`
# retains the optimization history of a sub-scenario after every execution.
# [get_result()][gemseo.scenario.mdo.MDOScenario.get_result] then gives access to a
# [BiLevelScenarioResult][gemseo.scenario.scenario_result.bilevel_scenario_result.BiLevelScenarioResult],
# whose
# [get_sub_scenario_history_dataset()][gemseo.scenario.scenario_result.bilevel_scenario_result.BiLevelScenarioResult.get_sub_scenario_history_dataset]
# combines all these histories into a single
# [Dataset][gemseo.dataset.dataset.Dataset] for analysis and plotting.
