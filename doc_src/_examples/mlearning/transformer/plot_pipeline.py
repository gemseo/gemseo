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
Pipeline
========
"""

from __future__ import annotations

from numpy import allclose
from numpy import linspace
from numpy import newaxis

from gemseo.mlearning.transformers.pipeline import Pipeline
from gemseo.mlearning.transformers.scaler.scaler import Scaler

# %%
# To illustrate the pipeline,
# we consider very simple data:
data = linspace(0, 1, 100)[:, newaxis]

# %%
# First,
# we create a pipeline of two transformers:
# the first one shifts the data while the second one reduces their amplitude.
pipeline = Pipeline(transformers=[Scaler(offset=1), Scaler(coefficient=2)])

# %%
# Then,
# we fit this :class:`.Pipeline` to the data, transform them and compute the Jacobian:
transformed_data = pipeline.fit_transform(data)
transformed_jac_data = pipeline.compute_jacobian(data)

# %%
# Lastly,
# we can do the same with two scalers:
shifter = Scaler(offset=1)
shifted_data = shifter.fit_transform(data)
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
# Note that a :class:`.Pipeline` can compute the Jacobian
# as long as the :class:`.BaseTransformer` instances that make it up can do so.
