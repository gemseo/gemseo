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
from gemseo.api import create_design_space
from gemseo.api import create_discipline
from gemseo.api import create_scenario
from gemseo.api import execute_post
from gemseo.api import get_available_opt_algorithms
from scipy import optimize

# PART 1


def f(x1=0.0, x2=0.0):
    y = x1 + x2
    return y


discipline = create_discipline("AutoPyDiscipline", py_func=f)

design_space = create_design_space()
design_space.add_variable("x1", l_b=-5, u_b=5, var_type="integer")
design_space.add_variable("x2", l_b=-5, u_b=5, var_type="integer")

scenario = create_scenario(
    discipline, "DisciplinaryOpt", "y", design_space, scenario_type="DOE"
)
scenario.execute({"algo": "fullfact", "n_samples": 11**2})

opt_results = scenario.get_optimum()
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
design_space.add_variable("x", l_b=-2.0, u_b=2.0, value=-0.5 * np.ones(1))

scenario = create_scenario(discipline, "DisciplinaryOpt", "y", design_space)
scenario.execute({"algo": "L-BFGS-B", "max_iter": 100})

opt_results = scenario.get_optimum()
print(
    "The solution of P is (x*,f(x*)) = ({}, {})".format(
        opt_results.x_opt, opt_results.f_opt
    )
)

algo_list = get_available_opt_algorithms()
print("Available algorithms:" + str(algo_list))

# POST TREATMENTS

problem = scenario.formulation.opt_problem
problem.export_hdf("my_optim.hdf5")
execute_post(problem, "OptHistoryView", save=True, file_path="opt_view_with_doe")
# or execute_post("my_optim.hdf5", "OptHistoryView", save=True,
# file_path="opt_view_from_disk")
