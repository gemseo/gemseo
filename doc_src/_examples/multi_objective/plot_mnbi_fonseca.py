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
r"""
Multi-objective Fonseca-Fleming example with the mNBI algorithm
===============================================================

In this example, the modified Normal Boundary Intersection algorithm (mNBI) is used to
solve the :class:`.FonsecaFleming` optimization problem :cite:`fonseca1995overview`:

.. math::

   \begin{aligned}
   \text{minimize the objective function }
   & f_1(x) = 1 - exp(-\sum_{i=1}^{d}((x_i - 1 / sqrt(d)) ^ 2)) \\
   & f_2(x) = 1 + exp(-\sum_{i=1}^{d}((x_i + 1 / sqrt(d)) ^ 2)) \\
   \text{with respect to the design variables }&x\\
   \text{subject to the bound constraint}
   & x\in[-4,4]^d
   \end{aligned}

We also show how the Pareto front can be refined.
"""

from __future__ import annotations

from gemseo import configure_logger
from gemseo import execute_algo
from gemseo import execute_post
from gemseo.algos.opt.mnbi.settings.mnbi_settings import MNBI_Settings
from gemseo.problems.multiobjective_optimization.fonseca_fleming import FonsecaFleming

configure_logger()


# %%
# Solve the Fonseca-Fleming optimization problem
# ----------------------------------------------
# The 3 sub-optimization problems of mNBI are solved with SLSQP,
# a gradient-based optimization algorithm from the NLOPT library,
# with a maximum of 100 iterations.
# The analytic gradients are provided.
opt_problem = FonsecaFleming()
mnbi_settings = MNBI_Settings(
    max_iter=1000,
    sub_optim_max_iter=100,
    n_sub_optim=3,
    sub_optim_algo="NLOPT_SLSQP",
)
result = execute_algo(opt_problem, settings_model=mnbi_settings)
# %%
# Display the Pareto front
# ^^^^^^^^^^^^^^^^^^^^^^^^
# |g| detects the Pareto optimal points and the dominated ones.
# The Fonseca-Fleming problem is interesting because
# its Pareto front is not convex.
# The mNBI algorithm successfully computes it.

execute_post(opt_problem, post_name="ParetoFront", save=False, show=True)

# %%
# Solve the Fonseca-Fleming optimization problem more finely
# ----------------------------------------------------------
# The Pareto front is then refined with 10 sub-optimizations instead of 3.
opt_problem = FonsecaFleming()
mnbi_settings.n_sub_optim = 10
result = execute_algo(opt_problem, settings_model=mnbi_settings)
# %%
# Display the Pareto front
# ^^^^^^^^^^^^^^^^^^^^^^^^
# We can clearly see the effect of the refinement.

execute_post(opt_problem, post_name="ParetoFront", save=False, show=True)
