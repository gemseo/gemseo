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
"""# Post-process an OptimizationDataset

## Problem

You have an [OptimizationDataset][gemseo.datasets.optimization_dataset.OptimizationDataset]
in memory and want to visualise the optimisation history
without going back to the original scenario or HDF5 file.

## Solution

Pass the [OptimizationDataset][gemseo.datasets.optimization_dataset.OptimizationDataset]
directly to [execute_post()][gemseo.execute_post],
just as you would pass a scenario or an HDF5 file path.
The dataset must have been built with `group_functions=True`
so that functions are correctly mapped to their optimisation role
(objective, constraints, observables).

## Step-by-step guide
"""

from __future__ import annotations

from gemseo import execute_post
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.settings.post import OptHistoryView_Settings

# %%
# ### 1. Build the dataset
#
# In this how-to, we assume that only an
# [OptimizationDataset][gemseo.datasets.optimization_dataset.OptimizationDataset]
# is available — no live scenario, no HDF5 file.
# As a workaround to create the prerequisite,
# we load an [OptimizationProblem][gemseo.algos.optimization_problem.OptimizationProblem]
# from an HDF5 file stored in the documentation directory
# and convert it to an [OptimizationDataset][gemseo.datasets.optimization_dataset.OptimizationDataset].
# The argument `group_functions=True` is required
# so that functions are grouped by their optimisation role:
problem = OptimizationProblem.from_hdf("power2_opt_pb.h5")
dataset = problem.to_dataset(group_functions=True)

# %%
# ### 2. Post-process the dataset
#
# Pass the dataset to [execute_post()][gemseo.execute_post]
# exactly as you would pass a scenario or an HDF5 file path:
execute_post(
    dataset,
    settings_model=OptHistoryView_Settings(
        save=False,
        show=True,
    ),
)

# %%
# ## Summary
#
# [execute_post()][gemseo.execute_post] accepts an
# [OptimizationDataset][gemseo.datasets.optimization_dataset.OptimizationDataset]
# as its first argument.
# The dataset must be built with `group_functions=True`
# to ensure functions are correctly mapped to their optimisation role.
