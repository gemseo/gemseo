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
a scenario orchestrates the execution of a set of disciplines to solve an optimization
or DOE problem.
However,
a scenario is not itself a discipline
since it cannot be directly plugged into another scenario as one of its components.
This becomes a problem when you need to compose scenarios hierarchically,
as the outer scenario cannot treat the inner one
as just another discipline in its workflow.

## Solution

You have to transform your scenario into a [Discipline][gemseo.core.discipline.discipline.Discipline],
using the
[MDOScenarioAdapter][gemseo.disciplines.scenario_adapters.mdo_scenario_adapter.MDOScenarioAdapter]
class.

## Step-by-step guide

Here,
the simple function $y = (x+1)^2 + n$ is minimized,
where $x \in \mathbb{R}$ and $n \in \mathbb{N}$.
First, a scenario is created to minimize $y$ w.r.t $x$.
Then, an upper-scenario is made to execute the first scenario as a sub-scenario.
"""

from __future__ import annotations

from numpy import array
from numpy import ones

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.pydoe.settings.pydoe_fullfact import PYDOE_FULLFACT_Settings
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.disciplines.scenario_adapters.mdo_scenario_adapter import MDOScenarioAdapter
from gemseo.formulations.disciplinary_opt_settings import DisciplinaryOpt_Settings
from gemseo.scenarios.mdo import MDOScenario
from gemseo.settings.opt import NLOPT_COBYLA_Settings

# %%
# ### 1. Define your scenario
#
# Here, the created scenario minimize $y$ w.r.t. $x$.

discipline = AnalyticDiscipline({"y": "(x+1)**2 + n"})

design_space = DesignSpace()
design_space.add_variable("x", lower_bound=-5, upper_bound=5.0, value=ones(1))

inner_scenario = MDOScenario(
    (discipline,), design_space, formulation_settings=DisciplinaryOpt_Settings()
)
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
#     The execution an
#     [MDOScenarioAdapter][gemseo.disciplines.scenario_adapters.mdo_scenario_adapter.MDOScenarioAdapter]
#     relies on the optimization of the inner scenario.
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
upper_scenario = MDOScenario(
    (scenario_adapter,),
    upper_design_space,
    formulation_settings=DisciplinaryOpt_Settings(),
)
upper_scenario.add_objective("y")
upper_scenario.execute(PYDOE_FULLFACT_Settings(n_samples=6))
upper_scenario.to_dataset()
# %%
# ## Summary
#
# You can transform an existing [MDOScenario][gemseo.scenarios.mdo.MDOScenario]
# into a
# [Discipline][gemseo.core.discipline.discipline.Discipline]
# with the
# [MDOScenarioAdapter][gemseo.disciplines.scenario_adapters.mdo_scenario_adapter.MDOScenarioAdapter].
#
# That way, you can simply create multi-level optimization processes.
