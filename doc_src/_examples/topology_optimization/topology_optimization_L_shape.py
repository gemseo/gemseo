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
Solve a 2D L-shape topology optimization problem
================================================
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from gemseo.api import configure_logger
from gemseo.api import create_scenario
from gemseo.problems.topo_opt.topopt_initialize import (
    initialize_design_space_and_discipline_to,
)
from matplotlib import colors

configure_logger()
# %%
# Setup the topology optimization problem
# ---------------------------------------
# Define the target volume fractio:
volume_fraction = 0.3

# %%
# Define the problem type:
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
# Generate a :class:`.MDOScenario`:
scenario = create_scenario(
    disciplines,
    formulation="DisciplinaryOpt",
    objective_name="compliance",
    design_space=design_space,
)
# %%
# Add the volume fraction constraint to the scenario:
scenario.add_constraint("volume fraction", "ineq", value=volume_fraction)

# %%
# Generate the XDSM
scenario.xdsmize()

# %%
# Execute the scenario
scenario.execute({"max_iter": 200, "algo": "NLOPT_MMA"})

# %%
# Results
# -------
# Post-process the optimization history:
scenario.post_process(
    "BasicHistory",
    variable_names=["compliance"],
    save=True,
    show=False,
    file_name=problem_name + "_history.png",
)

# %%
# .. image:: /_images/topology_optimization/L-Shape_history.png

# %%
# Plot the solution
plt.ion()  # Ensure that redrawing is possible
fig, ax = plt.subplots()
im = ax.imshow(
    -scenario.optimization_result.x_opt.reshape((n_x, n_y)).T,
    cmap="gray",
    interpolation="none",
    norm=colors.Normalize(vmin=-1, vmax=0),
)
fig.show()
im.set_array(-scenario.optimization_result.x_opt.reshape((n_x, n_y)).T)
fig.canvas.draw()
plt.savefig(problem_name + "_solution.png")

# %%
# .. image:: /_images/topology_optimization/L-Shape_solution.png
