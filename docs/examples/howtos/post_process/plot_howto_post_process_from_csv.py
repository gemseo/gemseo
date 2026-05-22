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
"""# Post-process a CSV file

## Problem

You have optimisation results stored in a CSV file
and want to visualise the history using GEMSEO post-processing algorithms,
without access to the original scenario or HDF5 file.

## Solution

Load the CSV as an
[OptimizationDataset][gemseo.datasets.optimization_dataset.OptimizationDataset],
then manually attach the
[OptimizationMetadata][gemseo.datasets.optimization_metadata.OptimizationMetadata]
that GEMSEO needs to run post-processing
(objective name, feasibility, optimum iteration, tolerances, etc.).
Optionally attach the input space for post-processors that need it.

## Step-by-step guide
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gemseo import execute_post
from gemseo.algos.constraint_tolerances import ConstraintTolerances
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.datasets.optimization_dataset import OptimizationDataset
from gemseo.datasets.optimization_metadata import OptimizationMetadata
from gemseo.settings.post import OptHistoryView_Settings

# %%
# ### 1. Create the CSV file
#
# In this how-to, we assume that only a CSV file is available.
# As a workaround to create the prerequisite,
# we load an [OptimizationProblem][gemseo.algos.optimization_problem.OptimizationProblem]
# from an HDF5 file stored in the documentation directory,
# convert it to an [OptimizationDataset][gemseo.datasets.optimization_dataset.OptimizationDataset],
# and save it as a CSV file:
problem = OptimizationProblem.from_hdf("power2_opt_pb.h5")
problem.to_dataset(group_functions=True).to_csv("results.csv")

# %%
# The file can be seen:
print(Path("results.csv").read_text())

# %%
# ### 2. Load the CSV file
#
# Load the CSV as an
# [OptimizationDataset][gemseo.datasets.optimization_dataset.OptimizationDataset]:
dataset = OptimizationDataset.from_csv("results.csv")

# %%
# ### 3. Build the optimization metadata
#
# When loading from a CSV, the metadata that GEMSEO normally derives from the
# optimisation problem must be provided manually.
#
# **Constraint name mapping** — maps each constraint output name to its
# associated constraint names. Here there are no name changes:
output_name_to_constraint_names = {
    name: [name]
    for name in dataset.inequality_constraint_names + dataset.equality_constraint_names
}

# %%
# **Optimum iteration** — the iteration at which the objective is minimal:
optimum_iteration = dataset.objective_dataset.idxmin(axis=0).values[0]

# %%
# **Tolerances** — using the default constraint tolerances:
tolerances = ConstraintTolerances()

# %%
# **Feasible iterations** — iterations where all constraints are satisfied
# within the given tolerances:
equality_feasible_mask = (
    np.abs(dataset.equality_constraint_dataset) <= tolerances.equality
).all(axis=1)
inequality_feasible_mask = (
    np.abs(dataset.inequality_constraint_dataset) <= tolerances.inequality
).all(axis=1)
feasible_iterations = dataset.index[
    equality_feasible_mask & inequality_feasible_mask
].tolist()

# %%
# ### 4. Attach the metadata to the dataset
#
# Create the
# [OptimizationMetadata][gemseo.datasets.optimization_metadata.OptimizationMetadata]
# and store it in the `misc` attribute of the dataset
# under the key `"optimization_metadata"`:
dataset.misc["optimization_metadata"] = OptimizationMetadata(
    objective_name="pow2",
    standardized_objective_name="pow2",
    minimize_objective=True,
    use_standardized_objective=False,
    tolerances=tolerances,
    output_name_to_constraint_names=output_name_to_constraint_names,
    feasible_iterations=feasible_iterations,
    optimum_iteration=optimum_iteration,
)

# %%
# ### 5. Attach the input space
#
# Some post-processors use the input space of the problem.
# Attach it to the dataset via the `misc` attribute:
input_space = DesignSpace()
input_space.add_variable("x", 3, lower_bound=-1.0, upper_bound=1.0, value=1.0)
dataset.misc["input_space"] = input_space

# %%
# ### 6. Post-process the dataset
#
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
# To post-process an
# [OptimizationDataset][gemseo.datasets.optimization_dataset.OptimizationDataset]
# loaded from a CSV file,
# you must manually build and attach an
# [OptimizationMetadata][gemseo.datasets.optimization_metadata.OptimizationMetadata]
# to `dataset.misc["optimization_metadata"]`,
# and optionally attach the input space to `dataset.misc["input_space"]`.
# Once done, pass the dataset to [execute_post()][gemseo.execute_post]
# as usual.
#
# !!! warning
#
#     The [GradientSensitivity][gemseo.post.gradient_sensitivity.GradientSensitivity]
#     post-processor with `compute_missing_gradients=True`
#     cannot be used with an
#     [OptimizationDataset][gemseo.datasets.optimization_dataset.OptimizationDataset].
