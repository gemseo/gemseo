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
r"""# Transform a scenario into a discipline

## Problem

In GEMSEO,
a scenario orchestrates the execution of a set of disciplines
to solve an optimization or evaluation problem.
However,
a scenario is not itself a discipline
since it cannot be directly plugged into another scenario as one of its components.
This becomes a problem when you need to compose scenarios hierarchically,
as the outer scenario cannot treat the inner one
as just another discipline in its workflow.

## Solution

You have to transform your scenario into a [Discipline][gemseo.core.discipline.discipline.Discipline],
using a scenario adapter.

GEMSEO provides two of them:

- [MDOScenarioAdapter][gemseo.scenario.adapter.mdo.MDOScenarioAdapter]
  wraps a scenario solving an
  [OptimizationProblem][gemseo.optimization.problem.OptimizationProblem],
  e.g. an [MDOScenario][gemseo.scenario.mdo.MDOScenario];
  its outputs are those of the optimum.
- [EvaluationScenarioAdapter][gemseo.scenario.adapter.evaluation.EvaluationScenarioAdapter]
  wraps any [EvaluationScenario][gemseo.scenario.evaluation.EvaluationScenario];
  its outputs are those of the last design point evaluated by the scenario,
  e.g. the last sample of a DOE algorithm.

[MDOScenarioAdapter][gemseo.scenario.adapter.mdo.MDOScenarioAdapter]
derives from
[EvaluationScenarioAdapter][gemseo.scenario.adapter.evaluation.EvaluationScenarioAdapter]
and adds the features requiring an optimum,
namely the Lagrange multipliers as extra outputs (`output_multipliers=True`),
the optimal objective value recorded by the scenario (`output_optimal_objective=True`)
and the linearization by post-optimal analysis.
It raises a `TypeError` at instantiation
when the scenario does not solve an
[OptimizationProblem][gemseo.optimization.problem.OptimizationProblem];
use
[EvaluationScenarioAdapter][gemseo.scenario.adapter.evaluation.EvaluationScenarioAdapter]
in that case.

## Step-by-step guide

Here,
the simple function $y = (x+1)^2 + n$ is minimized,
where $x \in \mathbb{R}$ and $n \in \mathbb{N}$.
First, a scenario is created to minimize $y$ w.r.t $x$.
Then, an upper-scenario is made to execute the first scenario as a sub-scenario.
Lastly, a scenario merely sampling $y$ w.r.t $x$ is transformed the same way.
"""

from __future__ import annotations

from numpy import array
from numpy import ones

from gemseo.discipline import AnalyticDiscipline
from gemseo.doe import PYDOE_FULLFACT_Settings
from gemseo.optimization import NLOPT_COBYLA_Settings
from gemseo.scenario import EvaluationScenario
from gemseo.scenario import EvaluationScenarioAdapter
from gemseo.scenario import MDOScenario
from gemseo.scenario import MDOScenarioAdapter
from gemseo.space import DesignSpace

# %%
# ### 1. Define your scenario
#
# Here, the created scenario minimize $y$ w.r.t. $x$.

discipline = AnalyticDiscipline({"y": "(x+1)**2 + n"})

design_space = DesignSpace()
design_space.add_variable("x", lower_bound=-5, upper_bound=5.0, value=ones(1))

inner_scenario = MDOScenario((discipline,), design_space)
inner_scenario.add_objective("y")
# %%
# ### 2. Set a default algorithm
#
# NLOPT_COBYLA is chosen to find the best $x$ to minimize $y$.
inner_scenario.set_algorithm(NLOPT_COBYLA_Settings(max_iter=100))

# %%
# ### 3. Transform your scenario
#
# When transforming a scenario into a discipline, the variable names must be defined.
#
# Here, the discipline has one input $n$, and returns both $x$ and $y$.
scenario_adapter = MDOScenarioAdapter(
    inner_scenario, input_names=["n"], output_names=["x", "y"]
)

# %%
# ### 4. Use your new discipline
#
# Now, `scenario_adapter` is now a discipline.
# You can either execute it for $n=4$:
scenario_adapter.execute({"n": array([4])})

# !!! note
#     The execution of an
#     [MDOScenarioAdapter][gemseo.scenario.adapter.mdo.MDOScenarioAdapter]
#     relies on the execution of the inner scenario.
#     This can been observed through log messages.
#
# or use another scenario to create a bi-level scenario:
upper_design_space = DesignSpace()
upper_design_space.add_variable(
    "n",
    lower_bound=-3,
    upper_bound=2.0,
    value=ones(1, dtype=int),
    type_=upper_design_space.DesignVariableType.INTEGER,
)
upper_scenario = MDOScenario((scenario_adapter,), upper_design_space)
upper_scenario.add_objective("y")
upper_scenario.execute(PYDOE_FULLFACT_Settings(n_samples=6))
upper_scenario.to_dataset()
# %%
# ### 5. Transform a scenario that does not optimize
#
# A scenario sampling $y$ w.r.t. $x$ does not solve an optimization problem,
# so it has no optimum to report
# and an
# [MDOScenarioAdapter][gemseo.scenario.adapter.mdo.MDOScenarioAdapter]
# would raise a `TypeError` at instantiation.
# Use an
# [EvaluationScenarioAdapter][gemseo.scenario.adapter.evaluation.EvaluationScenarioAdapter]
# instead:
sampling_design_space = DesignSpace()
sampling_design_space.add_variable("x", lower_bound=-5, upper_bound=5.0, value=ones(1))
sampling_scenario = EvaluationScenario(
    (AnalyticDiscipline({"y": "(x+1)**2 + n"}),), sampling_design_space
)
sampling_scenario.add_observable("y")
sampling_scenario.set_algorithm(PYDOE_FULLFACT_Settings(n_samples=5))

sampling_scenario_adapter = EvaluationScenarioAdapter(
    sampling_scenario, input_names=["n"], output_names=["x", "y"]
)
sampling_scenario_adapter.execute({"n": array([4])})

# %%
# Its output data are those of the last design point evaluated by the scenario,
# here the last sample of the full-factorial DOE,
# namely $x=5$ and so $y=(5+1)^2+4=40$:
sampling_scenario_adapter.output_data

# %%
# !!! note
#     The optimum-only features of the
#     [MDOScenarioAdapter][gemseo.scenario.adapter.mdo.MDOScenarioAdapter],
#     namely `output_multipliers`, `output_optimal_objective`
#     and the linearization by post-optimal analysis,
#     are not available for an
#     [EvaluationScenarioAdapter][gemseo.scenario.adapter.evaluation.EvaluationScenarioAdapter].

# %%
# ## Summary
#
# You can transform an existing [MDOScenario][gemseo.scenario.mdo.MDOScenario]
# into a
# [Discipline][gemseo.core.discipline.discipline.Discipline]
# with the
# [MDOScenarioAdapter][gemseo.scenario.adapter.mdo.MDOScenarioAdapter],
# and any [EvaluationScenario][gemseo.scenario.evaluation.EvaluationScenario]
# with the
# [EvaluationScenarioAdapter][gemseo.scenario.adapter.evaluation.EvaluationScenarioAdapter].
#
# That way, you can simply create multi-level optimization processes.
