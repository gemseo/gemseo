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
"""# DOE from existing samples

## Problem

You need to evaluate the functions at the existing points.
These points may be known from an external tool or a previous study, for example.

## Solution

Use [CustomDOE][gemseo.algos.doe.custom_doe.custom_doe.CustomDOE]
and pass your samples via the `samples` setting.

## Step-by-step guide
"""

from __future__ import annotations

import numpy as np

from gemseo.algos.design_space import DesignSpace
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.scenarios.evaluation import EvaluationScenario
from gemseo.settings import CustomDOE_Settings

# %%
# ### 1. Build the discipline and design space
#
discipline = AnalyticDiscipline({"y": "a*b + c"})

design_space = DesignSpace()
design_space.add_variable("a", lower_bound=1.0, upper_bound=10.0)
design_space.add_variable("b", lower_bound=1.0, upper_bound=10.0)
design_space.add_variable("c", size=2, lower_bound=1.0, upper_bound=10.0)

# %%
# ### 2. Define the samples
#
samples = [
    {
        "a": np.array([1.0]),
        "b": np.array([2.0]),
        "c": np.array([1.0, 1.0]),
    },
    {
        "a": np.array([2.0]),
        "b": np.array([3.0]),
        "c": np.array([5.0, 5.0]),
    },
]

# %%
# !!! note
#     A 2D-array can also be used to define the samples.
#     In that case,
#     the order of the elements must follow the design space variable order.
#     Use the
#     [variable_names][gemseo.algos.design_space.DesignSpace.variable_names]
#     method to see that order.
#
# ### 3. Execute the scenario with the custom samples
#
scenario = EvaluationScenario([discipline], design_space)
scenario.add_observable("y")
scenario.execute(CustomDOE_Settings(samples=samples))

# %%
# ### 4. Inspect the results
#
# Export the database to a [Dataset][gemseo.datasets.dataset.Dataset]
# and verify that the output equals the product of $a$ and $b$:
dataset = scenario.to_dataset(name="custom_sampling")
dataset

# %%
# ## Summary
#
# To evaluate functions at a predefined set of input points,
# you can either pass a NumPy array of shape `(n_samples, input_dimension)`
# or a list of dictionaries of 1D-arrays
# to `CustomDOE_Settings(samples=...)`.
