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
"""# Evaluate the functions outside the bounds of the design space

## Problem

You want to evaluate the functions of an optimization problem
at a point outside the bounds of its design space.
GEMSEO raises an error.

## Solution

Set the option `check_desvars_bounds` of the
[global configuration][concept-global-configuration] to `False`.

## Step-by-step guide
"""

from __future__ import annotations

from numpy import array
from numpy import cos
from numpy import sin

from gemseo import configuration
from gemseo.core.function.array_function import ArrayFunction
from gemseo.optimization import OptimizationProblem
from gemseo.space import DesignSpace

# %%
# ### Prerequisites
#
# An optimization problem whose design variable `x` is between 0 and 1:
design_space = DesignSpace()
design_space.add_real_variable("x", lower_bound=0.0, upper_bound=1.0, value=0.5)

problem = OptimizationProblem(design_space)
problem.objective = ArrayFunction(sin, name="f", jac=cos, expr="sin(x)")

# %%
# By default,
# evaluating the functions at `x=2` raises an error:
try:
    problem.evaluate_functions(array([2.0]), input_value_is_normalized=False)
except ValueError as error:
    print(error)

# %%
# ### 1. Disable the bound checks
configuration.check_desvars_bounds = False

# %%
# ### 2. Evaluate the functions outside the bounds
output_values, _ = problem.evaluate_functions(
    array([2.0]), input_value_is_normalized=False
)
output_values

# %%
# ## Summary
#
# Set `configuration.check_desvars_bounds` to `False`.
# An [OptimizationProblem][gemseo.optimization.problem.OptimizationProblem]
# then evaluates its functions outside the bounds of its design space.
# This also applies to the problems created before the change.
