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
"""
Transformer pipeline example
============================

In this example, we will create a pipeline of transformers.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from gemseo.api import configure_logger
from gemseo.mlearning.transform.pipeline import Pipeline
from gemseo.mlearning.transform.scaler.scaler import Scaler
from numpy import allclose
from numpy import linspace
from numpy import matmul
from numpy import sin

configure_logger()


# %%
# Create dataset
# --------------
x = linspace(0, 1, 100)[:, None]
data = sin(10 * x) - 3 * x


# %%
# Create transformer pipeline
# ---------------------------
# We create a pipeline of two transformers; the first performing a shift, the
# second a scale (both scalers). This could also be achieved using one scaler,
# but we here present a pipeline doing these transformations separately for
# illustrative purposes.
shift = Scaler(offset=5)
scale = Scaler(coefficient=0.5)
pipeline = Pipeline(transformers=[shift, scale])

# %%
# Transform data
# --------------
# In order to use the transformer, we have to fit it to the data.
pipeline.fit(data)

# %%
# Transform data using the pipeline
transformed_data = pipeline.transform(data)

# %%
# Transform data using individual components of the pipeline
only_shifted_data = shift.transform(data)

# %%
# Plot data
# ---------
plt.plot(x, data, label="Original data")
plt.plot(x, transformed_data, label="Shifted and scaled data")
plt.plot(x, only_shifted_data, label="Shifted but not scaled data")
plt.legend()
plt.show()

# %%
# Compute jacobian
# ----------------
jac = pipeline.compute_jacobian(data)
only_shift_jac = shift.compute_jacobian(data)
only_scale_jac = scale.compute_jacobian(only_shifted_data)

print(jac.shape)
print(only_shift_jac.shape)
print(only_scale_jac.shape)
print(allclose(jac, matmul(only_scale_jac, only_shift_jac)))
