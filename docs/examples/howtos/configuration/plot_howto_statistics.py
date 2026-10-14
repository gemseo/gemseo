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
"""# Count the executions of disciplines and the evaluations of functions

## Problem

You want to know how many times a discipline has been executed.
You also want to know how many times a function has been evaluated.

## Solution

Set the options `enable_discipline_statistics` and `enable_function_statistics`
of the [global configuration][concept-global-configuration] to `True`.

## Step-by-step guide
"""

from __future__ import annotations

from numpy import array
from numpy import cos
from numpy import sin

from gemseo import configuration
from gemseo import execute_algo
from gemseo.core.function.array_function import ArrayFunction
from gemseo.discipline import AnalyticDiscipline
from gemseo.optimization import L_BFGS_B_Settings
from gemseo.optimization import OptimizationProblem
from gemseo.space import DesignSpace

# %%
# ### 1. Enable the statistics
configuration.enable_discipline_statistics = True
configuration.enable_function_statistics = True

# %%
# ### 2. Count the executions of a discipline
discipline = AnalyticDiscipline({"y": "2*x"})
for x in (1.0, 2.0, 3.0):
    discipline.execute({"x": array([x])})

statistics = discipline.execution_statistics
statistics.n_executions, statistics.n_linearizations

# %%
# The total execution time in seconds is also recorded:
statistics.duration

# %%
# ### 3. Count the evaluations of a function
design_space = DesignSpace()
design_space.add_variable("x", lower_bound=-2.0, upper_bound=2.0, value=-0.5)

problem = OptimizationProblem(design_space)
problem.objective = ArrayFunction(sin, name="f", jac=cos)
execute_algo(problem, settings_model=L_BFGS_B_Settings(max_iter=10))

problem.objective.n_calls

# %%
# ## Summary
#
# Set `configuration.enable_discipline_statistics` to `True`.
# The attribute `execution_statistics` of a discipline then gives
# its number of executions, its number of linearizations and its execution time.
# Set `configuration.enable_function_statistics` to `True`.
# The attribute `n_calls` of a function then gives its number of evaluations.
#
# ## One step further
#
# The execution statistics can also be displayed as a Gantt chart,
# see [Execution statistics as a Gantt chart][].
