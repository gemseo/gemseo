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
"""# Post-process an HDF5 file

## Problem

After saving the results of an evaluation or optimisation to an HDF5 file,
you need to visualise the history or access the raw data without re-running
the problem.

## Solution

Use [execute_post()][gemseo.execute_post] to visualise the history directly
from the HDF5 file, and [import_database()][gemseo.import_database] to access
the raw data and convert it to a [Dataset][gemseo.datasets.dataset.Dataset].

## Step-by-step guide
"""

from __future__ import annotations

from gemseo import execute_post
from gemseo import import_database
from gemseo.post import BasicHistory_Settings

# %%
# ### 1. Visualise the evaluation history
#
# Pass the HDF5 file path and a post-processing settings model to
# [execute_post()][gemseo.execute_post].
# The file can come from an
# [EvaluationProblem][gemseo.algos.evaluation_problem.EvaluationProblem],
# an [OptimizationProblem][gemseo.algos.optimization_problem.OptimizationProblem]
# or an [MDOScenario][gemseo.scenarios.mdo.MDOScenario]:
execute_post("my_results.hdf5", BasicHistory_Settings(variable_names=["y_4"]))

# %%
# !!! tip "See also"
#     Different [post-processings][algorithms-for-post-processings] are available.
#
# !!! note
#
#     By default, GEMSEO saves the images on the disk.
#     Use `save=False` to not save figures and `show=True`
#     to display them on the screen.

# %%
# ### 2. Access the raw data
#
# Load the HDF5 file as a [Database][gemseo.algos.database.Database]:
database = import_database("my_results.hdf5")

# %%
# Then convert it to a [Dataset][gemseo.datasets.dataset.Dataset] for easier manipulation:
dataset = database.to_dataset()

# %%
# ## Summary
#
# Use [execute_post()][gemseo.execute_post] to visualise an HDF5 results file
# without re-running the problem, and [import_database()][gemseo.import_database]
# to access the raw data as a [Dataset][gemseo.datasets.dataset.Dataset].
#
# ## One step further
#
# - [Save a scenario for post-processing][save-a-scenario-for-post-processing]
# - [Save an optimization problem for post-processing][save-an-optimizationproblem-for-post-processing]
