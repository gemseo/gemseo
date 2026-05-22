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
"""# History backup

## Problem

You want to avoid losing all computed function evaluations
when a scenario execution is interrupted or crashes,
forcing costly recomputation from scratch.

## Solution

Use
[EvaluationScenario.set_backup_settings][gemseo.scenarios.evaluation.EvaluationScenario.set_backup_settings]
to write each function evaluation to an HDF5 file as the execution progresses.
On restart,
pass `load=True` to pre-load the backup into the database before the algorithm starts,
so that the already-computed points can be reused without re-evaluating the discipline.

## Step-by-step guide
"""

from __future__ import annotations

from pathlib import Path

from numpy import array

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt.scipy_local.settings.slsqp import SLSQP_Settings
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.scenarios.mdo import MDOScenario

BACKUP_FILE = Path("backup.hdf5")

# %%
# ### 1. Create the scenario
#
# For this optimization scenario,
# we minimize the sphere function $f(x, y) = x^2 + y^2$ over two variables,
# both bounded in $[-5, 5]$ with starting point $(4, 4)$.
# The global minimum is at the origin with $f = 0$.
discipline = AnalyticDiscipline({"obj": "x**2 + y**2"})

design_space = DesignSpace()
design_space.add_variable("x", lower_bound=-5.0, upper_bound=5.0, value=4.0)
design_space.add_variable("y", lower_bound=-5.0, upper_bound=5.0, value=4.0)

scenario = MDOScenario([discipline], design_space)
scenario.add_objective("obj")
# %%
# !!! note
#     This example also works with
#     [EvaluationScenario][gemseo.scenarios.evaluation.EvaluationScenario].
#
# ### 2. Configure the history backup
#
# [set_backup_settings][gemseo.scenarios.mdo.MDOScenario.set_backup_settings]
# registers a listener that appends each evaluation to an HDF5 file.
# Its key parameters are:
#
# - `file_path`: path to the HDF5 backup file.
# - `at_each_function_call`: write after every function call.
# - `at_each_iteration`: write after every algorithm iteration.
# - `erase`: delete the existing backup before the run.
# - `load`: pre-load the backup into the database before the run.
# - `plot`: save an optimization history view at each iteration (default: `False`,
#   [MDOScenario][gemseo.scenarios.mdo.MDOScenario] only).
#
# !!! warning
#     Passing both `erase=True` and `load=True` raises a `ValueError`
#     because they are mutually exclusive:
#     you cannot erase a file and then load it.
scenario.set_backup_settings(BACKUP_FILE)

# %%
# ### 3. Execute and verify the backup
#
# The backup file is created (or updated) incrementally during execution:
# by default the option `at_each_function_call` is used, therefore,
# every function call appends a new entry so the file is always consistent,
# even if the process is killed mid-run.
scenario.execute(SLSQP_Settings(max_iter=20))

# %%
# After execution we confirm that the file was written
# and check how many unique design points were stored in the database.
print(f"Backup file exists: {BACKUP_FILE.exists()}")
print(f"Evaluations stored: {len(scenario.formulation.problem.database)}")

# %%
# ### 4. Resume from backup
#
# Simulate a restart by resetting the starting point to the original value.
# In this case,
# this is necessary because the [DesignSpace][gemseo.algos.design_space.DesignSpace]
# retains the last visited point after the first run.
# This is not the case if you are relaunching your script for example.
design_space.set_current_value(array([4.0, 4.0]))

# %%
# Create the same scenario again and call
# [set_backup_settings][gemseo.scenarios.mdo.MDOScenario.set_backup_settings]
# with `load=True`.
# Before the algorithm starts, the backup is read and all previous evaluations
# are injected into the database.
scenario_2 = MDOScenario([discipline], design_space)
scenario_2.add_objective("obj")
scenario_2.set_backup_settings(BACKUP_FILE, load=True)

# %%
# The pre-loaded count matches the number of evaluations from the first run.
print(f"Pre-loaded evaluations: {len(scenario_2.formulation.problem.database)}")

# %%
# The scenario is re-executed:
scenario_2.execute(SLSQP_Settings(max_iter=20))

# %%
# While calling for the execution of `scenario_2`,
# there is no execution of the discipline since the database already covers
# all points the algorithm would request.
#
# !!! note
#     The algorithm is not aware that the database was pre-loaded.
#     It still proposes new candidates, but the
#     [Database][gemseo.algos.database.Database]
#     returns cached values instantly when a point is already known,
#     so no discipline evaluation takes place for those points.
#
# We remove the file for documentation purposes:
BACKUP_FILE.unlink()

# %%
# ## Summary
#
# [set_backup_settings][gemseo.scenarios.evaluation.EvaluationScenario.set_backup_settings]
# writes every function evaluation to an HDF5 file during the run.
# On restart,
# `load=True` pre-loads the backup so the algorithm reuses known evaluations
# without re-executing the discipline.
