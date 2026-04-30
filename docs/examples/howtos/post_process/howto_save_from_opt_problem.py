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
"""# Save an OptimizationProblem for post-processing

## Problem

After solving an
[OptimizationProblem][gemseo.algos.optimization_problem.OptimizationProblem],
you want to save the results to disk so they can be post-processed later
without re-running the optimisation.

## Solution

Call `to_hdf()` on the
[OptimizationProblem][gemseo.algos.optimization_problem.OptimizationProblem]
after solving it to export the results to an HDF5 file.

## Step-by-step guide
"""

from __future__ import annotations

from gemseo import execute_algo
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.core.functions.array_function import ArrayFunction

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

execute_algo(optimization_problem, "NLOPT_COBYLA", max_iter=10)

# %%
# ### 2. Save the results to an HDF5 file
#
optimization_problem.to_hdf("my_results.hdf")

# %%
# ## Summary
#
# Call `to_hdf()` on a solved
# [OptimizationProblem][gemseo.algos.optimization_problem.OptimizationProblem]
# to persist the results for later post-processing.
#
# ## One step further
#
# See [Post-process an HDF5 file][post-process-an-hdf5-file]
# to learn how to visualise the saved results.
