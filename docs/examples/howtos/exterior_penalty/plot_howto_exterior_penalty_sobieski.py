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
"""# Transform a constrained problem into an unconstrained one

## Problem

You want to solve a constrained optimisation problem
but the algorithm of your choice does not support constraints.

## Solution

Use apply_exterior_penalty() to transform the constrained problem
into an unconstrained one by folding the constraints into the objective function.

## Step-by-step guide
"""

from __future__ import annotations

from gemseo import create_discipline
from gemseo.problems.mdo.sobieski.core.design_space import SobieskiDesignSpace
from gemseo.scenarios.mdo import MDOScenario
from gemseo.settings import BasicHistory_Settings
from gemseo.settings import L_BFGS_B_Settings

# %%
# ### 1. Build the disciplines, design space and scenario
#
# We use the Sobieski SSBJ test case.
# See [the benchmark problems][benchmark-problems] for a full description.
disciplines = create_discipline([
    "SobieskiPropulsion",
    "SobieskiAerodynamics",
    "SobieskiMission",
    "SobieskiStructure",
])

design_space = SobieskiDesignSpace()

scenario = MDOScenario(disciplines, design_space)
scenario.add_objective("y_4", minimize=False)
scenario.set_differentiation_method()

# %%
# ### 2. Add the constraints
#
# The problem has three inequality constraints:
for constraint in ["g_1", "g_2", "g_3"]:
    scenario.add_constraint(constraint, constraint_type="ineq")

# %%
# ### 3. Apply the exterior penalty
#
# `apply_exterior_penalty()` removes the constraints from the problem
# and folds them into the objective function as a penalty term.
# After this call, the problem is unconstrained
# and can be passed to any gradient-based algorithm:
scenario.formulation.problem.apply_exterior_penalty(
    objective_scale=10.0, scale_inequality=10.0
)

# %%
# ### 4. Execute your scenario
#
# L-BFGS-B is a gradient-based algorithm that does not handle constraints.
# It can now be used safely on the penalised problem:
scenario.execute(L_BFGS_B_Settings(max_iter=10))

# %%
# ### 5. Inspect the constraint history
#
# Even though the constraints were folded into the objective,
# their history is still tracked and can be visualised:
scenario.post_process(
    BasicHistory_Settings(variable_names=["g_1", "g_2", "g_3"], save=False, show=True)
)

# %%
# ## Summary
#
# Call `apply_exterior_penalty()` on the formulation problem after adding constraints
# and before executing the scenario.
# This converts a constrained problem into an unconstrained one
# by penalising constraint violations in the objective function,
# making it compatible with algorithms like L-BFGS-B.
#
# ## One step further
#
# The quality of the approximation depends on the `objective_scale` and
# `scale_inequality` parameters: higher values yield a closer approximation
# of the original constrained problem but may require more iterations to converge.
