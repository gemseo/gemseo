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
Multi-objective Binh-Korn example with the mNBI algorithm
=========================================================

In this example, the modified Normal Boundary Intersection (mNBI) algorithm is used
to solve the :class:`.BinhKorn` problem :cite:`binh1997mobes`:

.. math::

   \begin{aligned}
   \text{minimize the objective function } & f_1(x, y) = 4x^2 + 4y^2 \\
   & f_2(x, y) = (x-5)^2 + (y-5)^2 \\
   \text{with respect to the design variables }&x,\,y \\
   \text{subject to the general constraints }
   & g_1(x,y) = (x-5)^2 + y^2 \leq 25.0\\
   & g_2(x, y) = (x-8)^2 + (y+3)^2 \geq 7.7\\
   \text{subject to the bound constraints }
   & 0 \leq x \leq 5.0\\
   & 0 \leq y \leq 3.0
   \end{aligned}

"""

from __future__ import annotations

from gemseo import configure_logger
from gemseo import execute_algo
from gemseo import execute_post
from gemseo.problems.analytical.binh_korn import BinhKorn

configure_logger()


# %%
# Solve the Binh-Korn optimization problem
# ----------------------------------------
# The 50 sub-optimization problems of mNBI are solved with SLSQP,
# a gradient-based optimization algorithm from the NLOPT library,
# with a maximum of 200 iterations.
# The analytic gradients are provided.
problem = BinhKorn()

result = execute_algo(
    problem,
    "MNBI",
    max_iter=10000,
    sub_optim_max_iter=200,
    n_sub_optim=50,
    sub_optim_algo="NLOPT_SLSQP",
)
# %%
# Display the Pareto front
# ------------------------
# |g| detects the Pareto optimal points and the dominated ones.
execute_post(problem, "ParetoFront", save=False, show=True)
