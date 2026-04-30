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
"""# Gradient sensitivity of objective and constraints

## Problem

After an optimisation, you want to visualise the sensitivity of the objective
and constraints to the design variables at the optimum,
using histograms of the gradients.

## Solution

Use [GradientSensitivity][gemseo.post.gradient_sensitivity.GradientSensitivity],
which plots histograms of the derivatives of the objective and constraints.
By default, the gradients are evaluated at the optimum
(or at the least infeasible point if the result is not feasible).

## Step-by-step guide
"""

from __future__ import annotations

from gemseo import create_discipline
from gemseo.formulations.mdf_settings import MDF_Settings
from gemseo.problems.mdo.sobieski.core.design_space import SobieskiDesignSpace
from gemseo.scenarios.mdo import MDOScenario
from gemseo.settings.mda import MDAGaussSeidel_Settings
from gemseo.settings.opt import SLSQP_Settings
from gemseo.settings.post import GradientSensitivity_Settings

# %%
# ### 1. Build and execute the scenario
#
# Unlike most post-processing how-tos, this one requires a **live scenario**
# rather than an HDF5 file.
# This is because computing missing gradients requires access to the underlying
# disciplines, which are not available when importing from an HDF5 file.
disciplines = create_discipline([
    "SobieskiPropulsion",
    "SobieskiAerodynamics",
    "SobieskiMission",
    "SobieskiStructure",
])

scenario = MDOScenario(
    disciplines,
    design_space=SobieskiDesignSpace(),
    formulation_settings=MDF_Settings(
        main_mda_settings=MDAGaussSeidel_Settings(
            max_mda_iter=30,
            tolerance=1e-10,
            warm_start=True,
            linear_solver_settings=None,
        ),
    ),
)
scenario.add_objective("y_4", minimize=False)

for name in ["g_1", "g_2", "g_3"]:
    scenario.add_constraint(name, constraint_type="ineq")

scenario.execute(SLSQP_Settings(max_iter=20))

# %%
# ### 2. Run the gradient sensitivity post-processing
#
# Set `compute_missing_gradients=True` to let GEMSEO compute gradients
# for any iteration where they were not evaluated by the algorithm.
scenario.post_process(
    GradientSensitivity_Settings(
        compute_missing_gradients=True,
        save=False,
        show=True,
    ),
)

# %%
# ## Summary
#
# [GradientSensitivity][gemseo.post.gradient_sensitivity.GradientSensitivity]
# plots histograms of the objective and constraint derivatives at the optimum.
# Use `compute_missing_gradients=True` to fill in gradients that the algorithm
# did not request — this requires a live scenario, not an HDF5 file.
# Use the `iteration` setting to compute sensitivities at a specific iteration
# instead of the optimum.
