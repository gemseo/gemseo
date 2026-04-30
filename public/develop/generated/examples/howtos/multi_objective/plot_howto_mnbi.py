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
#    INITIAL AUTHORS - initial API and implementation and/or initial documentation
#        :author:  Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""# Use the mNBI algorithm

## Problem

You have an optimization problem with more than one objective
and you want to find the solution
to the problem taking all objectives in consideration.

## Solution


When dealing with a multi-objective optimization problem,
a single optimal solution is meaningless — the goal is to compute the **Pareto front**
to gather enough trade-off information to make an informed choice
among all non-dominated solutions.

Use the [MNBI][gemseo.algos.opt.mnbi.mnbi.MNBI]
(modified Normal Boundary Intersection) algorithm.
It distributes $n$ sub-optimization problems uniformly along the boundary line
between the individual optima (anchor points) of each objective,
producing a well-spread Pareto front.

!!! tip
    GEMSEO provides ready-to-use benchmark multi-objective problems:
    [BinhKorn][gemseo.problems.multiobjective_optimization.binh_korn.BinhKorn],
    [FonsecaFleming][gemseo.problems.multiobjective_optimization.fonseca_fleming.FonsecaFleming],
    [Poloni][gemseo.problems.multiobjective_optimization.poloni.Poloni],
    and [Viennet][gemseo.problems.multiobjective_optimization.viennet.Viennet].

## Step-by-step guide
"""

from __future__ import annotations

from numpy import array

from gemseo import execute_algo
from gemseo import execute_post
from gemseo.algos.opt.nlopt.settings.nlopt_slsqp_settings import NLOPT_SLSQP_Settings
from gemseo.post import ParetoFront_Settings
from gemseo.problems.multiobjective_optimization.binh_korn import BinhKorn
from gemseo.settings.opt import MNBI_Settings

# %%
# ### 1. Define the optimization problem
#
# We use the
# [BinhKorn][gemseo.problems.multiobjective_optimization.binh_korn.BinhKorn]
# benchmark problem:
#
# $$
#    \begin{aligned}
#    \text{minimize} & \; f_1(x, y) = 4x^2 + 4y^2 \\
#                    & \; f_2(x, y) = (x-5)^2 + (y-5)^2 \\
#    \text{subject to} & \; g_1 = (x-5)^2 + y^2 \leq 25 \\
#                      & \; g_2 = (x-8)^2 + (y+3)^2 \geq 7.7 \\
#                      & \; 0 \leq x \leq 5,\quad 0 \leq y \leq 3
#    \end{aligned}
# $$
problem = BinhKorn()

# %%
# ### 2. Execute the mNBI algorithm
#
# The 50 sub-optimization problems are solved with SLSQP from NLOPT,
# with a maximum of 200 iterations each.
mnbi_settings = MNBI_Settings(
    max_iter=10000,
    n_sub_optim=50,
    sub_optim_algo_settings=NLOPT_SLSQP_Settings(max_iter=200),
)
_ = execute_algo(problem, settings_model=mnbi_settings)

# %%
# ### 3. Display the Pareto front
#
# GEMSEO detects the Pareto-optimal and dominated points automatically.
execute_post(problem, ParetoFront_Settings(save=False, show=True))

# %%
# ### 4. Refine the Pareto front in a specific area
#
# A region of interest can be refined by providing `custom_anchor_points`
# that bound both objectives in the target area.
# Here, 5 additional sub-optimizations zoom in on that region.
mnbi_settings = MNBI_Settings(
    max_iter=10000,
    n_sub_optim=5,
    sub_optim_algo_settings=NLOPT_SLSQP_Settings(max_iter=200),
    custom_anchor_points=[array([44.5, 14]), array([29.4, 19])],
)
_ = execute_algo(problem, settings_model=mnbi_settings)

# %%
# The refined region is now clearly visible in the updated Pareto front.
execute_post(problem, ParetoFront_Settings(save=False, show=True))

# %%
# ## Summary
#
# We used the [MNBI][gemseo.algos.opt.mnbi.mnbi.MNBI] algorithm to compute
# a well-spread Pareto front for the constrained
# [BinhKorn][gemseo.problems.multiobjective_optimization.binh_korn.BinhKorn] problem.
# The number of Pareto-optimal points is controlled by `n_sub_optim`.
# The front can be locally refined by providing `custom_anchor_points`.
