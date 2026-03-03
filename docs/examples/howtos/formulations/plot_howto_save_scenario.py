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
"""# Save the execution history of a scenario

## Problem

You have an [MDOScenario][gemseo.scenarios.mdo.MDOScenario] to execute.
For some reason, you want to store the evaluations to:

- Re-execute the scenario after a crash, without running points already evaluated,
- Keep the history of your scenario for archive.
- Post-process the convergence history.

## Solution

The execution history can be stored in an HDF5 file:
- during the execution
by setting a backup file with the
[set_backup_settings()][gemseo.scenarios.mdo.MDOScenario.set_backup_settings]
method before the execution,
- after the execution with the
[to_hdf()][gemseo.scenarios.mdo.MDOScenario.to_hdf]
or
[to_ggobi()][gemseo.scenarios.mdo.MDOScenario.to_ggobi]
methods when the execution is finished.

## Step-by-step guide
"""

from __future__ import annotations

from gemseo import create_discipline
from gemseo import create_scenario
from gemseo.algos.opt.scipy_local.settings.slsqp import SLSQP_Settings
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.problems.mdo.sobieski.core.design_space import SobieskiDesignSpace

# %%
# ### 1. Create your scenario
#
# Here, the Sobieski test case is used, with the MDF formulation.
disciplines = create_discipline([
    "SobieskiPropulsion",
    "SobieskiAerodynamics",
    "SobieskiMission",
    "SobieskiStructure",
])
design_space = SobieskiDesignSpace()
scenario = create_scenario(
    disciplines,
    "y_4",
    design_space,
    maximize_objective=True,
    formulation_name="MDF",
)

# %%
# ### 2. Set an history backup
#
# You can chose the save frequency,
# either at each iteration or/and at each call function.
# You can also erase first any existing backup file with the same name as the file name you provided,
# or decide to load an existing one to continue your scenario execution.
#
# !!! warning
#     This may slow down a lot the process execution.
#
# !!! tip
#     When discipline caches cannot be used on Windows due to multiprocessing limitations,
#     using the history backup is a good alternative.
scenario.set_backup_settings(
    file_path="mdf_backup.h5",
    at_each_iteration=True,
    at_each_function_call=False,
    erase=True,
    load=False,
)

# %%
# ### 3. Save after execution
scenario.execute(
    SLSQP_Settings(
        max_iter=10,
        ftol_rel=1e-10,
        ineq_tolerance=2e-3,
        normalize_design_space=True,
    )
)
scenario.to_hdf("mdf_history.h5")

# %%
# Different file formats are available:
[v.value for v in OptimizationProblem.HistoryFileFormat]

# %%
# ## Summary
#
# You can save an [MDOScenario][gemseo.scenarios.mdo.MDOScenario] history
# with [set_backup_settings()][gemseo.scenarios.mdo.MDOScenario.set_backup_settings]
# [to_hdf()][gemseo.scenarios.mdo.MDOScenario.to_hdf],
# or
# [to_ggobi()][gemseo.scenarios.mdo.MDOScenario.to_ggobi].
