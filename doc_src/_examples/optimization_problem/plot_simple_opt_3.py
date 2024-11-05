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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: François Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Analytical test case # 3
========================
"""

# %%
# In this example, we consider a simple optimization problem to illustrate
# algorithms interfaces and DOE libraries integration.
# Integer variables are used
#
# Imports
# -------
from __future__ import annotations

from numpy import sum as np_sum

from gemseo import configure_logger
from gemseo import execute_algo
from gemseo import execute_post
from gemseo import get_available_doe_algorithms
from gemseo import get_available_opt_algorithms
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.core.mdo_functions.mdo_function import MDOFunction

LOGGER = configure_logger()


# %%
# Define the objective function
# -----------------------------
# We define the objective function :math:`f(x)=\sum_{i=1}^dx_i`
# using an :class:`.MDOFunction`.
objective = MDOFunction(np_sum, name="f", expr="sum(x)")

# %%
# Define the design space
# -----------------------
# Then, we define the :class:`.DesignSpace` with |g|.
design_space = DesignSpace()
design_space.add_variable("x", 2, lower_bound=-5, upper_bound=5, type_="integer")

# %%
# Define the optimization problem
# -------------------------------
# Then, we define the :class:`.OptimizationProblem` with |g|.
problem = OptimizationProblem(design_space)
problem.objective = objective

# %%
# Solve the optimization problem using a DOE algorithm
# ----------------------------------------------------
# We can see this optimization problem as a trade-off
# and solve it by means of a design of experiments (DOE),
# e.g. full factorial design
execute_algo(problem, algo_name="PYDOE_FULLFACT", n_samples=11**2, algo_type="doe")

# %%
# Post-process the results
# ------------------------
execute_post(
    problem,
    post_name="ScatterPlotMatrix",
    variable_names=["x", "f"],
    save=False,
    show=True,
)

# %%
# Note that you can get all the optimization algorithms names:
get_available_opt_algorithms()
# %%
# and all the DOE algorithms names:
get_available_doe_algorithms()
