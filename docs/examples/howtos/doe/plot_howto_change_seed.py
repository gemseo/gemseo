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
"""# Ensure reproducibility in a DOE

## Problem

Stochastic DOE algorithms such as
[Latin hypercube sampling](https://en.wikipedia.org/wiki/Latin_hypercube_sampling)
produce different results each time they are executed,
because GEMSEO auto-increments the random seed at every execution.
This makes it difficult to retrieve or reproduce a specific set of samples.

## Solution

Pass an explicit `seed` value to the DOE settings to fix the random seed
and obtain the same samples across executions.

## Step-by-step guide
"""

from __future__ import annotations

from gemseo.algos.design_space import DesignSpace
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.scenarios.evaluation import EvaluationScenario
from gemseo.settings import OT_OPT_LHS_Settings

# %%
# ### 1. Build the discipline, design space and scenario
#
discipline = AnalyticDiscipline(expressions={"y": "x**2"})

design_space = DesignSpace()
design_space.add_variable("x", lower_bound=-1.0, upper_bound=1.0)

scenario = EvaluationScenario([discipline], design_space)
scenario.add_observable("y")
# %%
# ### 2. Observe the auto-increment behaviour
#
# By default, the seed starts at 0 and is incremented at each execution.
# Running the scenario twice therefore produces different samples:
scenario.execute(OT_OPT_LHS_Settings(n_samples=2))
scenario.formulation.problem.database.get_last_n_x_vect(2)

# %%
scenario.execute(OT_OPT_LHS_Settings(n_samples=2))
scenario.formulation.problem.database.get_last_n_x_vect(2)

# %%
# ### 3. Fix the seed for reproducibility
#
# Pass an explicit `seed` value to obtain the same samples every time:
scenario.execute(OT_OPT_LHS_Settings(n_samples=2, seed=0))
scenario.formulation.problem.database.get_last_n_x_vect(2)

# %%
# Running again with the same seed reproduces the exact same samples:
scenario.execute(OT_OPT_LHS_Settings(n_samples=2, seed=0))
scenario.formulation.problem.database.get_last_n_x_vect(2)

# %%
# ## Summary
#
# GEMSEO auto-increments the random seed at each execution of a stochastic DOE.
# To reproduce a specific set of samples, pass an explicit `seed` value
# to the DOE settings object.
