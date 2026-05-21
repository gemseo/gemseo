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

"""# Tutorial - Solve a 2D L-shape topology optimization problem

## Goal

In this tutorial, you will create your first topology optimization problem.
You will go through configuration to post-processing steps.
"""

from __future__ import annotations

from gemseo.problems.topology_optimization.topopt_initialize import (
    initialize_design_space_and_discipline_to,
)
from gemseo.scenarios.mdo import MDOScenario
from gemseo.settings.opt import NLOPT_MMA_Settings
from gemseo.settings.post import BasicHistory_Settings
from gemseo.settings.post import TopologyView_Settings

# %%
# ## Step 1 - Configure the topology optimization problem
#
# Define the target volume fraction:
volume_fraction = 0.3

# %%
# You can define the problem type.
# Here, you select the 2D L-Shape problem.
#
# !!! note
#     Other topology problems are available in GEMSEO,
#     such as `Short_Cantilever` and `MBB`.
problem_name = "L-Shape"

# %%
# Define the number of elements in the x- and y- directions:
n_x = 25
n_y = 25

# %%
# Define the full material Young's modulus and Poisson's ratio:
e0 = 1
nu = 0.3

# %%
# Define the penalty of the SIMP approach:
penalty = 3

# %%
# Define the minimum member size in the solution:
min_member_size = 1.5
# %%
# ## Step 2 - Create both disciplines and design space
#
# Simultaneously instantiate the [DesignSpace][gemseo.algos.design_space.DesignSpace]
# and the disciplines:
design_space, disciplines = initialize_design_space_and_discipline_to(
    problem=problem_name,
    n_x=n_x,
    n_y=n_y,
    e0=e0,
    nu=nu,
    penalty=penalty,
    min_member_size=min_member_size,
    vf0=volume_fraction,
)
# %%
# ## Step 3 - Create the topology optimization problem
#
# Generate an [MDOScenario][gemseo.scenarios.mdo.MDOScenario]:
scenario = MDOScenario(disciplines, design_space)

# %%
# Consider the optimization of the `compliance` objective:
scenario.add_objective("compliance")
# %%
# Add the volume fraction constraint to the scenario:
scenario.add_constraint(
    "volume fraction", constraint_type="ineq", value=volume_fraction
)

# %%
# ## Step 4 - Visualize your topology optimization problem
#
# You should always generate the XDSM before executing your scenario
scenario.xdsmize(save_html=False)

# %%
# Execute the scenario
scenario.execute(NLOPT_MMA_Settings(max_iter=200))

# %%
# ## Step 6 - Exploit the results
#
# You can see the convergence plot showing the decrease of the `compliance` variable,
# by using the `BasicHistory` plot.
scenario.post_process(
    BasicHistory_Settings(variable_names=["compliance"], show=True, save=False)
)

# %%
# And you can also visualize your optimized topology
# with the `TopologyView` post-process.
scenario.post_process(TopologyView_Settings(n_x=n_x, n_y=n_y, show=True, save=False))

# %%
# ## Key takeaways
#
# You've learnt to use GEMSEO to create a topology optimization problem.
# Such problem relies on the same GEMSEO basis:
# - create your disciplines,
# - create your [DesignSpace][gemseo.algos.design_space.DesignSpace],
# - create your [MDOScenario][gemseo.scenarios.mdo.MDOScenario].
#
# Then, you can exploit any wrapped solver that can handle your optimization problem,
# and post-process the results.
