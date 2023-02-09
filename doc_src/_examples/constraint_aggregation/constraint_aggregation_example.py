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

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.design_space import DesignVariableType
from gemseo.api import configure_logger
from gemseo.api import create_scenario
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.disciplines.concatenater import Concatenater

configure_logger()
# %%
# Number of constraints
N = 100

# %%
# Build the discipline
constraint_names = [f"g_{k + 1}" for k in range(N)]
function_names = ["o"] + constraint_names
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
    "x", l_b=0.0, u_b=1, value=1.0 / N / 2.0, var_type=DesignVariableType.FLOAT
)
ds.add_variable("y", l_b=0.0, u_b=1, value=1, var_type=DesignVariableType.FLOAT)

ds_new = deepcopy(ds)
# %%
# Build the optimization solver options
max_iter = 1000
ineq_tol = 1e-5
convergence_tol = 1e-8
normalize = True
algo_options = {
    "algo": "NLOPT_MMA",
    "max_iter": max_iter,
    "algo_options": {
        "ineq_tolerance": ineq_tol,
        "eq_tolerance": ineq_tol,
        "xtol_rel": convergence_tol,
        "xtol_abs": convergence_tol,
        "ftol_rel": convergence_tol,
        "ftol_abs": convergence_tol,
        "ctol_abs": convergence_tol,
        "normalize_design_space": normalize,
    },
}

# %%
# Build the optimization scenario
original_scenario = create_scenario(
    disciplines=[disc, concat],
    formulation="DisciplinaryOpt",
    objective_name="o",
    design_space=ds,
    maximize_objective=False,
)
original_scenario.add_constraint("g", "ineq")

original_scenario.execute(algo_options)
# Without constraint aggregation MMA iterations become more expensive, when a
# large number of constraints are activated.

# %%
# exploiting constraint aggregation on the same scenario:
new_scenario = create_scenario(
    disciplines=[disc, concat],
    formulation="DisciplinaryOpt",
    objective_name="o",
    design_space=ds_new,
    maximize_objective=False,
)
new_scenario.add_constraint("g", "ineq")

# %%
# This method aggregates the constraints using the KS function
new_scenario.formulation.opt_problem.aggregate_constraint(
    method="KS", rho=10.0, constr_id=0
)
new_scenario.execute(algo_options)

# %%
# with constraint aggregation the last iteration is faster.
