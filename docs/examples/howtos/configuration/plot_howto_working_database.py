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
"""# Record the evaluations in the coordinates of the algorithm

## Problem

The [database][concept-database] of a problem stores the evaluations
in the coordinates of its design space.
The algorithm may work in other coordinates.
You want to see the points as the algorithm sees them.

## Solution

Set the option `enable_working_database` of the
[global configuration][concept-global-configuration] to `True`
and read the `working_database` of the problem after the run.

## Step-by-step guide
"""

from __future__ import annotations

from numpy import cos
from numpy import sin

from gemseo import configuration
from gemseo import execute_algo
from gemseo.core.function.array_function import ArrayFunction
from gemseo.optimization import L_BFGS_B_Settings
from gemseo.optimization import OptimizationProblem
from gemseo.space import DesignSpace

# %%
# ### Prerequisites
#
# An optimization problem whose design variable `x` is between -2 and 2:
design_space = DesignSpace()
design_space.add_variable("x", lower_bound=-2.0, upper_bound=2.0, value=-0.5)

problem = OptimizationProblem(design_space)
problem.objective = ArrayFunction(sin, name="f", jac=cos)

# %%
# ### 1. Enable the working database
configuration.enable_working_database = True

# %%
# ### 2. Solve the problem
#
# The setting `normalize_design_space` of the algorithm is `True` by default.
# The algorithm then works in coordinates normalized between 0 and 1:
execute_algo(problem, settings_model=L_BFGS_B_Settings(max_iter=3))

# %%
# ### 3. Compare the two databases
#
# The database stores the points in the coordinates of the design space:
problem.database.to_dataset().get_view(variable_names="x")

# %%
# and the working database in the coordinates of the algorithm:
problem.working_database.to_dataset().get_view(variable_names="x")

# %%
# ## Summary
#
# Set `configuration.enable_working_database` to `True`.
# After the run, the `working_database` of the problem
# holds the evaluations in the coordinates of the algorithm.
# It is a debugging aid: nothing of GEMSEO reads it,
# and a run gives the same results with or without it.
