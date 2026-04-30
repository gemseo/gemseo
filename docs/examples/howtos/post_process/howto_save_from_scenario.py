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
"""# Save a scenario for post-processing

## Problem

After executing a scenario, you want to save the results to disk
so they can be post-processed later without re-running the optimisation.

## Solution

Call `to_hdf()` on the scenario after executing it
to export the results to an HDF5 file.

## Step-by-step guide
"""

from __future__ import annotations

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt.nlopt.settings.nlopt_cobyla_settings import NLOPT_COBYLA_Settings
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.formulations.disciplinary_opt_settings import DisciplinaryOpt_Settings
from gemseo.scenarios.mdo import MDOScenario

# %%
# ### 1. Build and execute the scenario
#
discipline = AnalyticDiscipline(expressions={"y": "x**2"})

design_space = DesignSpace()
design_space.add_variable("x", lower_bound=0.0, upper_bound=1.0)

scenario = MDOScenario(
    [discipline], design_space, formulation_settings=DisciplinaryOpt_Settings()
)
scenario.add_objective("y")
scenario.execute(NLOPT_COBYLA_Settings(max_iter=10))

# %%
# ### 2. Save the results to an HDF5 file
#
scenario.to_hdf("my_results.hdf")

# %%
# ## Summary
#
# Call `to_hdf()` on an executed scenario
# to persist the results for later post-processing.
#
# ## One step further
#
# See [Post-process an HDF5 file][post-process-an-hdf5-file]
# to learn how to visualise the saved results.
