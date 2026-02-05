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

"""# How to reduce a design space

## Problem

When working with optimization problems,
you may encounter design spaces with numerous variables.
Rather than creating a new design space from scratch, you want to reduce
an existing design space by removing unwanted variables or dimensions.

## Solution

GEMSEO provides methods to filter / remove variables:

## Step-by-step guide

You will learn three different approaches to:

- Remove a specific variable
- Keep only a subset of variables
- Filter specific dimensions of a variable

"""

from __future__ import annotations

from numpy import array
from numpy import ones

from gemseo import create_design_space

# %%
# ### 1. Create an initial design space
#
# First,
# let's create a design space with multiple variables of different types and sizes.
# This will serve as our starting point for demonstrating the reduction methods.

design_space = create_design_space()
design_space.add_variable("x1", value=array([1.0]))
design_space.add_variable("x2", value=array([1]), type_="integer")
design_space.add_variable("x3", size=2, value=ones(2))
design_space.add_variable("x4", lower_bound=ones(1), value=ones(1))
design_space.add_variable("x5", upper_bound=ones(1), value=ones(1))
design_space.add_variable("x6", value=ones(1))
design_space.add_variable(
    "x7",
    size=2,
    type_="integer",
    value=array([0, 1]),
    lower_bound=-ones(2),
    upper_bound=ones(2),
)

print("Initial design space:")
design_space

# %%
# ### 2. Remove a specific variable
#
# Use the `remove_variable()` method to delete a single variable from the design space.
# Here, we remove the variable `'x4'`.

design_space.remove_variable("x4")
print("After removing 'x4':")
design_space

# %%
# ### 3. Keep only selected variables
#
# The `filter()` method allows you to keep only a subset of variables.
# All other variables will be removed from the design space.
# Here, we keep only `'x1'`, `'x2'`, `'x3'`, and `'x6'`.

design_space.filter(["x1", "x2", "x3", "x6"])
print("After filtering to keep only selected variables:")
design_space

# %%
# ### 4. Filter dimensions of a variable
#
# For multi-dimensional variables,
# use `filter_dimensions()` to keep only specific components.
# Here, we keep only the first component (index 0) of the variable `'x3'`,
# reducing it from a 2D to a 1D variable.

design_space.filter_dimensions("x3", [0])
print("After filtering dimensions of 'x3':")
design_space

# %%
# ## Summary
#
# - **`remove_variable(name)`**: Removes a single variable by name
# - **`filter(names)`**: Keeps only the specified variables, removes all others
# - **`filter_dimensions(name, indices)`**: Keeps only the specified dimensions of a variable
#
# !!! tip
#     These methods modify the design space in place. If you need to preserve the original,
#     consider creating a copy before applying filters using `design_space.copy()`.
#
