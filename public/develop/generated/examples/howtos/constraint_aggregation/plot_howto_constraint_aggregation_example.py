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
"""# How to aggregate constraints

## Problem

When an optimisation problem involves a large number of inequality constraints,
each iteration becomes expensive because the solver must evaluate all of them.
Constraint aggregation reduces this cost by replacing the full set of constraints
with a single aggregated scalar constraint.

## Solution

After adding the constraints to the scenario,
call `constraints.aggregate()` on the formulation problem
to replace them with an aggregated scalar constraint
before executing the scenario.

## Step-by-step guide
"""

from __future__ import annotations

from gemseo import create_scenario
from gemseo.algos.design_space import DesignSpace
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.disciplines.constraint_aggregation import ConstraintAggregation
from gemseo.settings.opt import NLOPT_MMA_Settings

# %%
# ### 1. Build your scenario
#
# Define 100 inequality constraints:
N = 100
constraint_names = [f"g_{k + 1}" for k in range(N)]
function_names = ["o", *constraint_names]
function_expressions = ["y"] + [f"{k + 1}*x*exp(1-{k + 1}*x)-y" for k in range(N)]
disc = AnalyticDiscipline(
    dict(zip(function_names, function_expressions, strict=False)),
    name="function",
)

# %%
# Create your design space
ds = DesignSpace()
ds.add_variable(
    "x",
    lower_bound=0.0,
    upper_bound=1,
    value=1.0 / N / 2.0,
    type_=DesignSpace.DesignVariableType.FLOAT,
)
ds.add_variable(
    "y",
    lower_bound=0.0,
    upper_bound=1,
    value=1,
    type_=DesignSpace.DesignVariableType.FLOAT,
)

# %%
# And build the scenario with its 100 constraints
scenario = create_scenario(
    [disc],
    "o",
    ds,
    maximize_objective=False,
    formulation_name="DisciplinaryOpt",
)
scenario.add_constraint(constraint_names, constraint_type="ineq")

# %%
# ### 2. Aggregate the constraints
#
# Replace the 100 individual constraints with a single scalar one
# using the lower bound KS function:
scenario.formulation.problem.constraints.aggregate(
    0, method=ConstraintAggregation.EvaluationFunction.LOWER_BOUND_KS, rho=10.0
)

# %%
# !!! note
#     With the `group` argument, you can chose to aggregate constraints by group.
#
# ### 3. Execute the scenario
#
# The scenario now contains only 1 constraint:
scenario.execute(
    NLOPT_MMA_Settings(
        ineq_tolerance=1e-5,
        eq_tolerance=1e-5,
        xtol_rel=1e-8,
        xtol_abs=1e-8,
        ftol_rel=1e-8,
        ftol_abs=1e-8,
        normalize_design_space=True,
        max_iter=1000,
    )
)

# %%
# ## Summary
#
# Use `constraints.aggregate()` after `add_constraint()` and before `execute()`
# to replace a large set of inequality constraints with a scalar KS one,
# reducing the cost of each solver iteration.
#
# ## One step further
#
# Other aggregation methods are available besides `"lower_bound_KS"`,
# all stored into the
# [EvaluationFunction][gemseo.disciplines.constraint_aggregation.ConstraintAggregation.EvaluationFunction]
# class.
# The `rho` parameter controls the tightness of the approximation:
# a higher value gives a closer approximation but may cause numerical issues.
