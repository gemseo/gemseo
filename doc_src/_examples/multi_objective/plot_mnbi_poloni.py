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
Multi-objective Poloni example with the mNBI algorithm
======================================================

In this example,
the modified Normal Boundary Intersection (mNBI) algorithm is used to
solve the :class:`.Poloni` problem :cite:`POLONI2000403`:

.. math::

    \begin{aligned}
    &a1 = 0.5 * sin(1) - 2 * cos(1) + sin(2) - 1.5 * cos(2)\\
    &a2 = 1.5 * sin(1) - cos(1) + 2 * sin(2) - 0.5 * cos(2)\\
    &b1(x, y) = 0.5 * sin(x) - 2 * cos(x) + sin(y) - 1.5 * cos(y)\\
    &b2(x, y) = 1.5 * sin(x) - cos(x) + 2 * sin(y) - 0.5 * cos(y)\\
    \text{minimize the objective function}\\
    & f_1(x, y) = 1 + (a1 - b1(x,y)^2 + (a2 - b2(x,y))^2 \\
    & f_2(x, y) = (x + 3)^2 + (y + 1)^2 \\
    \text{with respect to the design variables }\\
    &x \\
    \text{subject to the bound constraints }\\
    & -\pi \leq x \leq \pi\\
    & -\pi \leq y \leq \pi
    \end{aligned}

It is an interesting one because its Pareto front is discontinuous.
"""

from __future__ import annotations

from gemseo import configure_logger
from gemseo import execute_algo
from gemseo import execute_post
from gemseo.algos.opt.mnbi.settings.mnbi_settings import MNBI_Settings
from gemseo.problems.multiobjective_optimization.poloni import Poloni

configure_logger()


# %%
# Solve the Poloni optimization problem
# -------------------------------------
# The 10 sub-optimization problems of mNBI are solved with SLSQP,
# a gradient-based optimization algorithm from the NLOPT library,
# with a maximum of 50 iterations.
# The analytic gradients are provided.
opt_problem = Poloni()
algo_settings = MNBI_Settings(
    max_iter=10000,
    sub_optim_max_iter=50,
    n_sub_optim=50,
    sub_optim_algo="SLSQP",
)
result = execute_algo(opt_problem, settings_model=algo_settings)

# %%
# Display the Pareto front
# ------------------------
# |g| detects the Pareto optimal points and the dominated ones.
# There is one interesting area that has a hole in the Pareto front.
# The mNBI algorithm avoids running sub-optimizations in this area
# which saves computation time.
execute_post(opt_problem, post_name="ParetoFront", save=False, show=True)
