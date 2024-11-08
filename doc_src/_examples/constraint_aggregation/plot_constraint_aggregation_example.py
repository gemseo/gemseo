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
"""
Examples for constraint aggregation
===================================
"""

from __future__ import annotations

from copy import deepcopy

from gemseo import configure_logger
from gemseo import create_scenario
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt.nlopt.settings.nlopt_mma_settings import NLOPT_MMA_Settings
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.disciplines.concatenater import Concatenater

configure_logger()
# %%
# Number of constraints
N = 100

# %%
# Build the discipline
constraint_names = [f"g_{k + 1}" for k in range(N)]
function_names = ["o", *constraint_names]
function_expressions = ["y"] + [f"{k + 1}*x*exp(1-{k + 1}*x)-y" for k in range(N)]
disc = AnalyticDiscipline(
    name="function",
    expressions=dict(zip(function_names, function_expressions)),
)
# This step is required to put all constraints needed for aggregation in one variable.
concat = Concatenater(constraint_names, "g")

# %%
# Build the design space
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

ds_new = deepcopy(ds)
# %%
# Build the optimization solver settings
mma_settings = NLOPT_MMA_Settings(
    ineq_tolerance=1e-5,
    eq_tolerance=1e-5,
    xtol_rel=1e-8,
    xtol_abs=1e-8,
    ftol_rel=1e-8,
    ftol_abs=1e-8,
    normalize_design_space=True,
    max_iter=1000,
)

# %%
# Build the optimization scenario
original_scenario = create_scenario(
    [disc, concat],
    "o",
    ds,
    maximize_objective=False,
    formulation_name="DisciplinaryOpt",
)
original_scenario.add_constraint("g", constraint_type="ineq")

original_scenario.execute(mma_settings)
# Without constraint aggregation MMA iterations become more expensive, when a
# large number of constraints are activated.

# %%
# exploiting constraint aggregation on the same scenario:
new_scenario = create_scenario(
    [disc, concat],
    "o",
    ds_new,
    maximize_objective=False,
    formulation_name="DisciplinaryOpt",
)
new_scenario.add_constraint("g", constraint_type="ineq")

# %%
# This method aggregates the constraints using the lower bound KS function
new_scenario.formulation.optimization_problem.constraints.aggregate(
    0, method="lower_bound_KS", rho=10.0
)
new_scenario.execute(mma_settings)

# %%
# with constraint aggregation the last iteration is faster.
