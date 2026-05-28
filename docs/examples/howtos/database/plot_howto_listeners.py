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
"""# Observe evaluations using listeners

## Problem

During a scenario execution, you may want to observe each evaluation
— track convergence, log design vectors, or trigger side effects —
without modifying the discipline or the algorithm.

## Solution

Use
[EvaluationProblem.add_listener()][gemseo.algos.evaluation_problem.EvaluationProblem.add_listener]
to register a callback that is called automatically by the
[Database][gemseo.algos.database.Database]
whenever new values are stored during the run.

## Step-by-step guide
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt.scipy_local.settings.slsqp import SLSQP_Settings
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.scenarios.mdo import MDOScenario

if TYPE_CHECKING:
    from gemseo.algos.optimization_problem import OptimizationProblem


# %%
# ### 0. Logging setup
#
# We set up the logging configuration to see the listener's output in the console.
# In this case we use GEMSEO's default configuration.
LOGGER = logging.getLogger("gemseo")


# %%
# ### 1. Create the scenario
#
# For this optimization scenario,
# We minimize the sphere function $f(x, y) = x^2 + y^2$ over two variables,
# both bounded in $[-5, 5]$ with starting point $(4, 4)$.
discipline = AnalyticDiscipline({"obj": "x**2 + y**2"})

design_space = DesignSpace()
design_space.add_variable("x", lower_bound=-5.0, upper_bound=5.0, value=4.0)
design_space.add_variable("y", lower_bound=-5.0, upper_bound=5.0, value=4.0)

scenario = MDOScenario([discipline], design_space)
scenario.add_objective("obj")

# %%
# ### 2. Define a listener
#
# A listener is any callable that accepts a single argument:
# the design vector `x_vect` at which new outputs were just stored.
# Inside the listener, use
# [Database.get_function_value][gemseo.algos.database.Database.get_function_value]
# to retrieve the corresponding output value.
#
# Here we track the objective at each iteration by capturing a reference to
# the [OptimizationProblem][gemseo.algos.optimization_problem.OptimizationProblem]
# in a closure; additionally, we log the design vector to demonstrate the live
# tracking of the process:
problem: OptimizationProblem = scenario.formulation.problem
obj_history = []


def track_objective(x_vect):
    value = problem.database.get_function_value("obj", x_vect)
    obj_history.append(float(value))
    LOGGER.info("Listener: x_vector = %s, obj = %s", x_vect, value)


# %%
# ### 3. Register the listener
#
# Pass the callback to
# [add_listener()][gemseo.algos.evaluation_problem.EvaluationProblem.add_listener].
# The default `at_each_iteration=True` fires it once per complete design point
# — the right choice for convergence tracking.
problem.add_listener(track_objective)

# %%
# ### 4. Execute and inspect the tracked history
scenario.execute(SLSQP_Settings(max_iter=20))

# %%
# As you can see in the log output, we see that at each iteration we properly
# capture the design vector and the corresponding objective value.
#
# After execution, `obj_history` holds one entry per iteration,
# showing the objective converging toward zero.
# These values can be observed in the log as well.
print(f"Listener called {len(obj_history)} times")
for i, val in enumerate(obj_history, 1):
    print(f"  Iteration {i}: obj = {val:.6f}")

# %%
# ## Summary
#
# [EvaluationProblem.add_listener()][gemseo.algos.evaluation_problem.EvaluationProblem.add_listener]
# attaches a callback to the [Database][gemseo.algos.database.Database].
# The callback receives the design vector and can query the database for output values.
#
# ## One step further
#
# For fine-grained control, the lower-level
# [Database.add_store_listener][gemseo.algos.database.Database.add_store_listener]
# and
# [Database.add_new_iter_listener][gemseo.algos.database.Database.add_new_iter_listener]
# are available directly on the database.
# Use [Database.clear_listeners][gemseo.algos.database.Database.clear_listeners]
# to remove callbacks when they are no longer needed.
