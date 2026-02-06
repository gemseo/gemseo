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
#                         documentation
#        :author: Syver Doving Agdestein
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""# Chain data transformations

## Problem

I would like to chain data transformations, e.g. scaling then dimension reduction.

How can I implement this pipeline?

## Solution

Create a [Pipeline][gemseo.machine_learning.transformers.pipeline.Pipeline]
of [BaseTransformer][gemseo.machine_learning.transformers.base_transformer.BaseTransformer]s.

## Step-by-step guide
"""

from __future__ import annotations

from numpy import allclose
from numpy import linspace
from numpy import newaxis

from gemseo.machine_learning.transformers.pipeline import Pipeline
from gemseo.machine_learning.transformers.scaler.scaler import Scaler

# %%
# ### 1. Generate data
#
# To illustrate the concept of pipeline,
# we consider very simple data:
data = linspace(0, 1, 100)[:, newaxis]

# %%
# ### 2. Create the pipeline
#
# Here,
# we want a pipeline that:
# 1. shifts the data by 1,
# 2. reduces their amplitude by 2.
pipeline = Pipeline(transformers=[Scaler(offset=1), Scaler(coefficient=2)])

# %%
# ### 3. Fit the pipeline to data
pipeline.fit(data)

# %%
# ### 4. Transform data
transformed_data = pipeline.transform(data)

# %%
# ### 5. Compute the Jacobian of the pipeline
transformed_jac_data = pipeline.compute_jacobian(data)

# %%
# ### 6. Verify the pipeline
#
# We could implement this pipeline by hand,
# with a shifter:
shifter = Scaler(offset=1)
shifted_data = shifter.fit_transform(data)
# %%
# and a scaler from the shifted data:
scaler = Scaler(coefficient=2)
data_shifted_then_scaled = scaler.fit_transform(shifted_data)
jac_shifted_then_scaled = scaler.compute_jacobian(
    shifted_data
) @ shifter.compute_jacobian(data)

# %%
# and verify that the results are identical:
assert allclose(transformed_data, data_shifted_then_scaled)
assert allclose(transformed_jac_data, jac_shifted_then_scaled)

# %%
# ## Summary
#
# Data transformations can be easily chained
# using a [Pipeline][gemseo.machine_learning.transformers.pipeline.Pipeline]
# of [BaseTransformer][gemseo.machine_learning.transformers.base_transformer.BaseTransformer]s.
# If the transformers are differentiable, the pipeline is differentiable too.
