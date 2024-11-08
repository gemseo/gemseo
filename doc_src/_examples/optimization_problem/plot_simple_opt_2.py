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
#        :author: François Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Analytical test case # 2
========================
"""

# %%
# In this example, we consider a simple optimization problem to illustrate
# algorithms interfaces and optimization libraries integration.
#
# Imports
# -------
from __future__ import annotations

from numpy import cos
from numpy import exp
from numpy import sin

from gemseo import configure_logger
from gemseo import execute_algo
from gemseo import execute_post
from gemseo import get_available_opt_algorithms
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.core.mdo_functions.mdo_function import MDOFunction

configure_logger()


# %%
# Define the objective function
# -----------------------------
# We define the objective function :math:`f(x)=\sin(x)-\exp(x)`
# using an :class:`.MDOFunction` defined by the sum of :class:`.MDOFunction` objects.
f_1 = MDOFunction(sin, name="f_1", jac=cos, expr="sin(x)")
f_2 = MDOFunction(exp, name="f_2", jac=exp, expr="exp(x)")
objective = f_1 - f_2

# %%
# .. seealso::
#
#    The following operators are implemented: addition, subtraction and multiplication.
#    The minus operator is also defined.
#
# Define the design space
# -----------------------
# Then, we define the :class:`.DesignSpace` with |g|.
design_space = DesignSpace()
design_space.add_variable("x", lower_bound=-2.0, upper_bound=2.0, value=-0.5)

# %%
# Define the optimization problem
# -------------------------------
# Then, we define the :class:`.OptimizationProblem` with |g|.
problem = OptimizationProblem(design_space)
problem.objective = objective

# %%
# Solve the optimization problem using an optimization algorithm
# --------------------------------------------------------------
# Finally, we solve the optimization problems with |g| interface.
#
# Solve the problem
# ^^^^^^^^^^^^^^^^^
optimization_result = execute_algo(problem, algo_name="L-BFGS-B")
optimization_result

# %%
# Note that you can get all the optimization algorithms names:
get_available_opt_algorithms()

# %%
# Save the optimization results
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We can serialize the results for further exploitation.
problem.to_hdf("my_optim.hdf5")

# %%
# Post-process the results
# ^^^^^^^^^^^^^^^^^^^^^^^^
execute_post(problem, post_name="OptHistoryView", save=False, show=True)

# %%
# .. note::
#
#    We can also save this plot using the arguments ``save=False``
#    and ``file_path='file_path'``.

# %%
# Solve the optimization problem using a DOE algorithm
# ----------------------------------------------------
# We can also see this optimization problem as a trade-off problem
# and solve it by means of a design of experiments (DOE).
problem.reset()
optimization_result = execute_algo(
    problem, algo_name="PYDOE_LHS", n_samples=10, algo_type="doe"
)
optimization_result
