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
"""# How to create a dataset from a NumPy array

## Problem

You have data stored in a NumPy array and want to load it into a
[Dataset][gemseo.datasets.dataset.Dataset]
with meaningful variable names and group assignments.

## Solution

Use the class method
[Dataset.from_array()][gemseo.datasets.dataset.Dataset.from_array],
passing the array along with optional variable names, sizes and group names.

## Step-by-step guide

"""

from __future__ import annotations

from numpy import concatenate
from numpy.random import default_rng

from gemseo.datasets.dataset import Dataset

# %%
# ### 1. Generate the data
#
# Let us consider three parameters $x_1$, $x_2$ and $x_3$
# of size 1, 2 and 3 respectively.
# We generate 5 random samples of the inputs where
#
# - x_1 is stored in the first column,
# - x_2 is stored in the 2nd and 3rd columns
#
# and 5 random samples of the outputs $x_3$:

rng = default_rng(1)

n_samples = 5
inputs = rng.random((n_samples, 3))
outputs = rng.random((n_samples, 3))
data = concatenate((inputs, outputs), axis=1)
data.shape

# %%
# !!! note
#     `data` should be of shape (n_samples, n_variables).
#
# ### 2. Create a dataset with default names
#
# Pass the array alone and let GEMSEO assign default column names:
dataset = Dataset.from_array(data)
dataset

# %%
# ### 3. Create a dataset with custom variable names and sizes
#
# Pass the variable names and a dictionary mapping each name to its number of components.
# The total number of components must equal the number of columns of the array:
name_to_size = {"x_1": 1, "x_2": 2, "y_1": 3}
dataset = Dataset.from_array(data, ["x_1", "x_2", "y_1"], name_to_size)
dataset

# %%
# !!! warning
#
#     The number of variable names must equal the number of columns of the array,
#     or a sizes dictionary must be provided whose component total matches that count.

# %%
# ### 4. Create a dataset with custom groups
#
# Pass an additional dictionary mapping each variable name to its group.
# Variables not listed fall back to
# [Dataset.DEFAULT_GROUP][gemseo.datasets.dataset.Dataset.DEFAULT_GROUP]:
groups = {"x_1": "inputs", "x_2": "inputs", "y_1": "outputs"}
dataset = Dataset.from_array(data, ["x_1", "x_2", "y_1"], name_to_size, groups)
dataset

# %%
# ## Summary
#
# [Dataset.from_array()][gemseo.datasets.dataset.Dataset.from_array]
# accepts up to four arguments in increasing order of specificity:
# the data array, variable names, a name-to-size mapping, and a name-to-group mapping.
# All arguments beyond the array are optional;
# omitted names, sizes or groups are filled in with GEMSEO defaults.
#
# ## One step further
#
# To read data directly from a file instead of a NumPy array, two class methods are available:
#
# - [Dataset.from_csv()][gemseo.datasets.dataset.Dataset.from_csv]
#   for CSV files structured with 3 header rows encoding the multi-index
#   (group, variable, component), or
# - [Dataset.from_txt()][gemseo.datasets.dataset.Dataset.from_txt]
#   for plain text files that contain no header rows.
