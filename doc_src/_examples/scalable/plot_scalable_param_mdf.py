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
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Parametric scalable MDO problem - MDF
=====================================
We define
a :class:`~.gemseo.problems.scalable.parametric.scalable_problem.ScalableProblem`
with a shared design variable of size 1
and 2 strongly coupled disciplines.
The first one has a local design variable of size 1
and a coupling variable of size 2
while the second one has a local design variable of size 3
and a coupling variable of size 4.

We would like to solve this MDO problem by means of an MDF formulation.
"""

from __future__ import annotations

from gemseo import configure_logger
from gemseo import execute_algo
from gemseo import execute_post
from gemseo import generate_n2_plot
from gemseo.problems.scalable.parametric.core.scalable_discipline_settings import (
    ScalableDisciplineSettings,
)
from gemseo.problems.scalable.parametric.scalable_problem import ScalableProblem

configure_logger()

# %%
# Instantiation of the scalable problem
# -------------------------------------
problem = ScalableProblem(
    [ScalableDisciplineSettings(1, 2), ScalableDisciplineSettings(3, 4)], 1
)

# %%
# Display the coupling structure
# ------------------------------
generate_n2_plot(problem.disciplines, save=False, show=True)

# %%
# Solve the MDO using an MDF formulation
# --------------------------------------
scenario = problem.create_scenario()
scenario.execute({"algo": "NLOPT_SLSQP", "max_iter": 100})

# %%
# Post-process the results
# ------------------------
scenario.post_process("OptHistoryView", save=False, show=True)

# %%
# Solve the associated quadratic programming problem
# --------------------------------------------------
problem = problem.create_quadratic_programming_problem()
execute_algo(problem, algo_name="NLOPT_SLSQP", max_iter=100)

# %%
# Post-process the results
# ------------------------
execute_post(problem, "OptHistoryView", save=False, show=True)
