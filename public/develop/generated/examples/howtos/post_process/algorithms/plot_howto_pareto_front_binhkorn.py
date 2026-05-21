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
"""# Pareto front on a multi-objective problem

## Problem

For a multi-objective problem, you want to visualise the Pareto front
and control whether non-feasible points are displayed.

## Solution

Use [ParetoFront][gemseo.post.pareto_front.ParetoFront] directly on the
optimisation problem after running a DOE.
Use `show_non_feasible` to toggle the display of non-feasible points
and `objectives_labels` to assign readable labels to the objectives.

## Step-by-step guide
"""

from __future__ import annotations

from gemseo import execute_post
from gemseo.algos.doe.factory import DOE_LIBRARY_FACTORY
from gemseo.algos.doe.openturns.settings.ot_opt_lhs import OT_OPT_LHS_Settings
from gemseo.post import ParetoFront_Settings
from gemseo.problems.multiobjective_optimization.binh_korn import BinhKorn

# %%
# ### 1. Build and sample the optimisation problem
#
# We use the Binh and Korn problem
# (see [BinhKorn][gemseo.problems.multiobjective_optimization.binh_korn.BinhKorn])
# and sample it with an optimised LHS:
problem = BinhKorn()
DOE_LIBRARY_FACTORY.execute(problem, OT_OPT_LHS_Settings(n_samples=100))

# %%
# ### 2. Plot the Pareto front without non-feasible points
#
# Set `show_non_feasible=False` to restrict the plot to feasible points only.
# Use `objectives_labels` to assign readable labels to the objective components:
execute_post(
    problem,
    ParetoFront_Settings(
        show_non_feasible=False,
        objectives=["compute_binhkorn"],
        objectives_labels=["f1", "f2"],
        save=False,
        show=True,
    ),
)

# %%
# ### 3. Plot the Pareto front with non-feasible points
#
# By default, non-feasible points are included and shown in green:
execute_post(
    problem,
    ParetoFront_Settings(
        objectives=["compute_binhkorn"],
        objectives_labels=["f1", "f2"],
        save=False,
        show=True,
    ),
)

# %%
# ## Summary
#
# [ParetoFront][gemseo.post.pareto_front.ParetoFront] can be applied directly
# to an optimisation problem after a DOE.
# Use `show_non_feasible=False` to hide non-feasible points
# and `objectives_labels` to assign readable names to the objective components.
