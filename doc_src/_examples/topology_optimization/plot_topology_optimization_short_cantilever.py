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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Simone Coniglio
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Solve a 2D short cantilever topology optimization problem
=========================================================
"""

from __future__ import annotations

from gemseo import configure_logger
from gemseo import create_scenario
from gemseo.problems.topology_optimization.topopt_initialize import (
    initialize_design_space_and_discipline_to,
)

configure_logger()
# %%
# Setup the topology optimization problem
# ---------------------------------------
# Define the target volume fraction:
volume_fraction = 0.3

# %%
# Define the problem type:
problem_name = "Short_Cantilever"

# %%
# Define the number of elements in the x- and y- directions:
n_x = 50
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
# Instantiate the :class:`.DesignSpace` and the disciplines:

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
# Solve the topology optimization problem
# ---------------------------------------
# Generate an :class:`.MDOScenario`:
scenario = create_scenario(
    disciplines,
    "compliance",
    design_space,
    formulation_name="DisciplinaryOpt",
)

# %%
# Add the volume fraction constraint to the scenario:
scenario.add_constraint(
    "volume fraction", constraint_type="ineq", value=volume_fraction
)

# %%
# Generate the XDSM:
scenario.xdsmize(save_html=False)

# %%
# Execute the scenario:
scenario.execute(algo_name="NLOPT_MMA", max_iter=200)

# %%
# Results
# -------
# Post-process the optimization history:
scenario.post_process(
    post_name="BasicHistory", variable_names=["compliance"], show=True, save=False
)

# %%
# Plot the solution
scenario.post_process(post_name="TopologyView", n_x=n_x, n_y=n_y, show=True, save=False)
