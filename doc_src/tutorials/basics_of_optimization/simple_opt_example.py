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
from __future__ import annotations

import numpy as np
from scipy import optimize

from gemseo import create_design_space
from gemseo import create_discipline
from gemseo import create_scenario
from gemseo import execute_post
from gemseo import get_available_opt_algorithms

# PART 1


def f(x1=0.0, x2=0.0):
    y = x1 + x2
    return y


discipline = create_discipline("AutoPyDiscipline", py_func=f)

design_space = create_design_space()
design_space.add_variable("x1", lower_bound=-5, upper_bound=5, type_="integer")
design_space.add_variable("x2", lower_bound=-5, upper_bound=5, type_="integer")

scenario = create_scenario(
    discipline,
    "y",
    design_space,
    scenario_type="DOE",
    formulation_name="DisciplinaryOpt",
)
scenario.execute(algo_name="PYDOE_FULLFACT", n_samples=11**2)

opt_results = scenario.optimization_result
print(f"The solution of P is (x*, f(x*)) = ({opt_results.x_opt}, {opt_results.f_opt})")


# PART 2


def g(x=0):
    y = np.sin(x) - np.exp(x)
    return y


def dgdx(x=0):
    y = np.cos(x) - np.exp(x)
    return y


x_0 = -0.5 * np.ones(1)
opt = optimize.fmin_l_bfgs_b(g, x_0, fprime=dgdx, bounds=[(-0.2, 2.0)])
x_opt, f_opt, _ = opt
print(f"The solution of P is (x*, f(x*)) = ({x_opt[0]}, {f_opt[0]})")


# PART 3

discipline = create_discipline("AutoPyDiscipline", py_func=g, py_jac=dgdx)

design_space = create_design_space()
design_space.add_variable(
    "x", lower_bound=-2.0, upper_bound=2.0, value=-0.5 * np.ones(1)
)

scenario = create_scenario(
    discipline, "y", design_space, formulation_name="DisciplinaryOpt"
)
scenario.execute(algo_name="L-BFGS-B", max_iter=100)

opt_results = scenario.optimization_result
print(f"The solution of P is (x*,f(x*)) = ({opt_results.x_opt}, {opt_results.f_opt})")

algo_list = get_available_opt_algorithms()
print("Available algorithms:" + str(algo_list))

# POST TREATMENTS

problem = scenario.formulation.optimization_problem
problem.to_hdf("my_optim.hdf5")
execute_post(problem, "OptHistoryView", save=True, file_path="opt_view_with_doe")
# or execute_post("my_optim.hdf5", "OptHistoryView", save=True,
# file_path="opt_view_from_disk")
