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
"""# Post-process an OptimizationProblem

## Problem

You have solved an
[OptimizationProblem][gemseo.algos.optimization_problem.OptimizationProblem]
directly (without a scenario) and want to visualise the history.

## Solution

Pass the
[OptimizationProblem][gemseo.algos.optimization_problem.OptimizationProblem]
directly to [execute_post()][gemseo.execute_post],
just as you would pass a scenario or an HDF5 file path.

## Step-by-step guide
"""

from __future__ import annotations

from gemseo import execute_algo
from gemseo import execute_post
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.core.functions.array_function import ArrayFunction
from gemseo.post import BasicHistory_Settings
from gemseo.settings import NLOPT_COBYLA_Settings

# %%
# ### 1. Build and solve the optimization problem
#
# We consider a minimization problem over the interval $[0,1]$
# of the $f(x)=x^2$ objective function:

objective = ArrayFunction(
    lambda x: x**2, name="f", input_names=["x"], output_names=["y"]
)

design_space = DesignSpace()
design_space.add_variable("x", lower_bound=0.0, upper_bound=1.0)

optimization_problem = OptimizationProblem(design_space)
optimization_problem.objective = objective

execute_algo(
    optimization_problem, "opt", settings_model=NLOPT_COBYLA_Settings(max_iter=10)
)

# %%
# ### 2. Post-process the optimization problem
#
execute_post(
    optimization_problem,
    BasicHistory_Settings(variable_names=["x"], save=False, show=True),
)

# %%
# !!! note
#
#     By default, GEMSEO saves the images on the disk.
#     Use `save=False` to not save figures and `show=True` to display them on the screen.

# %%
# ## Summary
#
# [execute_post()][gemseo.execute_post] accepts an
# [OptimizationProblem][gemseo.algos.optimization_problem.OptimizationProblem]
# as its first argument, alongside a scenario or an HDF5 file path.
