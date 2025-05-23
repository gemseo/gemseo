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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Jean-Christophe Giret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Pareto front on a multi-objective problem
=========================================

In this example, we illustrate the use of the :class:`.ParetoFront` plot
on the Binh and Korn multi-objective problem.
"""

from __future__ import annotations

from gemseo import configure_logger
from gemseo.algos.doe.factory import DOELibraryFactory
from gemseo.post.factory import PostFactory
from gemseo.problems.multiobjective_optimization.binh_korn import BinhKorn

# %%
# Import
# ------
# The first step is to import a high-level function for logging.

configure_logger()


# %%
# Import the optimization problem
# -------------------------------
# Then, we instantiate the Binh and Korn optimization problem (see :class:`.BinhKorn`).

problem = BinhKorn()

# %%
# Create and execute scenario
# ---------------------------
# Then,
# we instantiate the design of experiment factory,
# and we request the execution of a 100-length LHS optimized by simulated annealing.
doe_factory = DOELibraryFactory()
doe_factory.execute(problem, algo_name="OT_OPT_LHS", n_samples=100)

# %%
# Post-processing
# ---------------
# Lastly, we post-process the scenario by means of the :class:`.ParetoFront`
# plot which generates a plot or a matrix of plots if there are more than
# 2 objectives, plots in blue the locally non dominated points for the current
# two objectives, plots in green the globally (all objectives) Pareto optimal
# points. The plots in green denote non-feasible points. Note that the user
# can avoid the display of the non-feasible points.

PostFactory().execute(
    problem,
    post_name="ParetoFront",
    show_non_feasible=False,
    objectives=["compute_binhkorn"],
    objectives_labels=["f1", "f2"],
    save=False,
    show=True,
)

PostFactory().execute(
    problem,
    post_name="ParetoFront",
    objectives=["compute_binhkorn"],
    objectives_labels=["f1", "f2"],
    save=False,
    show=True,
)
