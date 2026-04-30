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

# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: François Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""# Define and solve an optimization problem

## Problem

You need to solve a pure optimization problem — no discipline,
just functions returning an array from an array.

## Solution

Use [OptimizationProblem][gemseo.algos.optimization_problem.OptimizationProblem]
together with [ArrayFunction][gemseo.core.functions.array_function.ArrayFunction]
to define the problem, then call [execute_algo()][gemseo.execute_algo]
to solve it.

## Step-by-step guide
"""

from __future__ import annotations

from numpy import cos
from numpy import exp
from numpy import sin

from gemseo import execute_algo
from gemseo import execute_post
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt.scipy_local.settings.lbfgsb import L_BFGS_B_Settings
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.core.functions.array_function import ArrayFunction
from gemseo.post import OptHistoryView_Settings

# %%
# ### 1. Define the objective function
#
# We wrap functions returning a NumPy array from a NumPy array into [ArrayFunctions][gemseo.core.functions.array_function.ArrayFunction], and compose these array functions with arithmetic operators.
# Here we build the objective function $f(x) = \sin(x) - \exp(x)$ with its analytic Jacobian:

f = ArrayFunction(sin, name="f", jac=cos, expr="sin(x)")
g = ArrayFunction(exp, name="g", jac=exp, expr="exp(x)")
objective = f - g

# %%
# !!! note
#     The `expr` argument is used by the string representation of the function,
#     e.g. `str(f) == "f = sin(x)"` and `str(objective) == "[f+g] = sin(x)+exp(x)"`.
#
# !!! info "See also"
#     [ArrayFunction][gemseo.core.functions.array_function.ArrayFunction]
#     supports addition, subtraction, and multiplication of functions.
#     The unary minus operator is also available.

# %%
# ### 2. Define the design space
#
# The [DesignSpace][gemseo.algos.design_space.DesignSpace]
# sets the variable bounds and the initial point:

design_space = DesignSpace()
design_space.add_variable("x", lower_bound=-2.0, upper_bound=2.0, value=-0.5)

# %%
# ### 3. Assemble the optimization problem
#
# [OptimizationProblem][gemseo.algos.optimization_problem.OptimizationProblem]
# holds the objective, constraints (none here), and the design space:

problem = OptimizationProblem(design_space)
problem.objective = objective

# %%
# ### 4. Solve with a gradient-based optimizer
#
# [execute_algo()][gemseo.execute_algo] runs any GEMSEO-registered algorithm
# on the problem and returns an
# [OptimizationResult][gemseo.algos.optimization_result.OptimizationResult]:

optimization_result = execute_algo(problem, settings_model=L_BFGS_B_Settings())
optimization_result

# %%
# !!! note
#     This problem could have been solved by a DOE, such as `PYDOE_LHS`.
#
# Save the full optimization history to an HDF5 file for later post-processing:

problem.to_hdf("my_optim.hdf5")

# %%
# Visualize the convergence history with
# [OptHistoryView][gemseo.post.opt_history_view.OptHistoryView]:

execute_post(problem, OptHistoryView_Settings(save=False, show=True))

# %%
# !!! note
#     [execute_post()][gemseo.execute_post] also accepts the HDF5 file path
#     directly: `execute_post("my_optim.hdf5", ...)`.
#
# ## Summary
#
# [OptimizationProblem][gemseo.algos.optimization_problem.OptimizationProblem]
# is the low-level way to define an optimization problem
# without any discipline or formulation.
# Pass it to [execute_algo()][gemseo.execute_algo] with any optimizer or DOE algorithm.
# Use [problem.to_hdf()][gemseo.algos.optimization_problem.OptimizationProblem.to_hdf]
# to persist results and [execute_post()][gemseo.execute_post]
# to visualize the optimization history.
