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
"""# Tutorial - From a monodisciplinary problem to an MDO problem

## Goal

MDO benchmark problems are far less numerous than standard optimization problems.
This tutorial shows how
[OptAsMDOScenario][gemseo.problems.mdo.opt_as_mdo_scenario.OptAsMDOScenario]
automatically transforms any monodisciplinary optimization problem into a
multidisciplinary one, introducing strongly coupled disciplines and coupling variables
without changing the optimal solution.
We will learn what design variable renaming means,
how to read the resulting coupling graph,
and how the MDO solution compares to the original.

!!! quote "References"

    - Matthias De Lozzo, Olivier Roustant and Amine Aziz-Alaoui,
      Make an optimization problem multidisciplinary.
      Preprint.
      URL: <https://arxiv.org/abs/2512.19217>.
    - Amine Aziz-Alaoui.
      Contributions to multidisciplinary design optimization under uncertainty,
      with applications to aircraft design.
      Theses, Université de Toulouse, February 2025.
      URL: <https://theses.hal.science/tel-05059696>.
"""

from __future__ import annotations

from numpy import array

from gemseo import generate_coupling_graph
from gemseo.algos.design_space import DesignSpace
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.problems.mdo.opt_as_mdo_scenario import OptAsMDOScenario
from gemseo.scenarios.mdo import MDOScenario
from gemseo.settings.opt import NLOPT_SLSQP_Settings

# %%
# ## Step 1 — Define the discipline and design space
#
# We use the 3-dimensional Rosenbrock function as our test problem:
#
# $$f(z) = 100(z_2-z_1^2)^2 + (1-z_1)^2 + 100(z_1-z_0^2)^2 + (1-z_0)^2$$
#
# Its unique minimizer over $[-1,1]^3$ is $z^*=(1,1,1)$ with $f(z^*)=0$.
# `AnalyticDiscipline` is the lightest way to wrap a symbolic expression in GEMSEO:
# derivatives are computed automatically via symbolic differentiation.
discipline = AnalyticDiscipline(
    {"f": "100*(z_2-z_1**2)**2+(1-z_1)**2+100*(z_1-z_0**2)**2+(1-z_0)**2"},
    name="Rosenbrock",
)

design_space = DesignSpace()
design_space.add_variable("z_0", lower_bound=-1, upper_bound=1)
design_space.add_variable("z_1", lower_bound=-1, upper_bound=1)
design_space.add_variable("z_2", lower_bound=-1, upper_bound=1)
# %%
# The starting point for the optimizers is
# $x^{(0)}=(-0.25, 0.75, -0.9)$.
initial_point = array([-0.25, 0.75, -0.9])
design_space.set_current_value(initial_point)

# %%
# ## Step 2 — Solve the monodisciplinary baseline
#
# Before introducing MDO, we solve the original single-discipline problem.
# This gives us a reference solution to compare against later.
opt_scenario = MDOScenario([discipline], design_space)
opt_scenario.add_objective("f")
opt_scenario.execute(NLOPT_SLSQP_Settings(max_iter=100))
opt_scenario.optimization_result

# %%
# ## Step 3 — Build the MDO scenario with OptAsMDOScenario
#
# [OptAsMDOScenario][gemseo.problems.mdo.opt_as_mdo_scenario.OptAsMDOScenario]
# requires at least three scalar inputs and one output.
# It splits the design variables across strongly coupled disciplines automatically.
# The only choice to make here is the MDO formulation;
# we use the default one, namely MDF.
design_space.set_current_value(initial_point)

mdo_scenario = OptAsMDOScenario(discipline, design_space)
mdo_scenario.add_objective("f")

# %%
# ## Step 4 — Understand the variable renaming
#
# [OptAsMDOScenario][gemseo.problems.mdo.opt_as_mdo_scenario.OptAsMDOScenario]
# renames the original design variables according to a fixed
# convention to reflect their role in the MDO problem:
#
# - $x_0$ is the **global** design variable (shared across all disciplines),
# - $x_{1+i}$ is the **local** design variable owned by the $i$-th strongly coupled
#   discipline.
#
design_space

# %%
# ## Step 5 — Read the coupling graph
#
# The coupling graph reveals the MDO structure that `OptAsMDOScenario` has built.
# Two strongly coupled disciplines $D_1$ and $D_2$ exchange coupling variables
# $y_1$ and $y_2$, which together with the design variables $x_0, x_1, x_2$
# encode the original problem in MDO form.
# Both disciplines are weakly coupled downstream to a link discipline $L$,
# which reconstructs the original variables $z_0, z_1, z_2$ from
# the MDO design and coupling variables before passing them to the Rosenbrock discipline.
generate_coupling_graph(mdo_scenario.disciplines, file_path="")

# %%
# ## Step 6 — Solve the MDO problem and compare
#
# The result should match the monodisciplinary baseline: $z^*=(1,1,1)$, $f=0$.
mdo_scenario.execute(NLOPT_SLSQP_Settings(max_iter=100))
mdo_scenario.optimization_result

# %%
# ## Key takeaways
#
# - [OptAsMDOScenario][gemseo.problems.mdo.opt_as_mdo_scenario.OptAsMDOScenario]
#   turns any monodisciplinary problem (≥3 scalar inputs,
#   ≥1 output) into an MDO problem with strongly coupled disciplines, with no
#   change to the optimal solution.
# - Design variables are renamed: $x_0$ is global, $x_{1+i}$ is local to
#   discipline $i$.
# - A link discipline $L$ reconstructs the original variables from the MDO ones,
#   keeping the Rosenbrock discipline untouched.
