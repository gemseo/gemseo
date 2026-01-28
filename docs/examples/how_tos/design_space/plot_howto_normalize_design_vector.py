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
#                           documentation
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""# How to (un)normalize design parameters

## Problem

You want to use normalization on your design variables.

## Solution

The design space has two methods to normalize variables:

- [normalize_vect()][gemseo.algos.design_space.DesignSpace.normalize_vect]
- [unnormalize_vect()][gemseo.algos.design_space.DesignSpace.unnormalize_vect]

## Step-by-step guide
"""

from __future__ import annotations

from numpy import array
from numpy import ones

from gemseo import create_design_space

# %%
# ### 1. Create a design space
#
# First, let's create a design space.
design_space = create_design_space()
design_space.add_variable("x1", lower_bound=-10, upper_bound=10)
design_space.add_variable("x2", lower_bound=-10, upper_bound=10)
design_space.add_variable("x3", lower_bound=-10, upper_bound=10)
design_space.add_variable("x4", value=ones(1), lower_bound=-10, upper_bound=10)

# %%
# ### 2. Normalize a given array

normalized_x_vect = design_space.normalize_vect(array([1.0, 10.0, 1.0, 1.0]))
normalized_x_vect

# %%
# !!! note
#     When each variable has a current value, it can be retrieved as its normalized form with:
#     ``design_space.get_current_value(normalize=True)``.
#
# ### 3. Un-normalize the array
unnormalized_x_vect = design_space.unnormalize_vect(normalized_x_vect)
unnormalized_x_vect

# %%
# !!! note
#     Both methods takes an optional argument denoted `'minus_lb'`
#     which is `True` by default.
#     If `True`,
#     the normalization of the normalizable variables is of the form
#     `(x-lb_x)/(ub_x-lb_x)`.
#     Otherwise,
#     it is of the form `x/(ub_x-lb_x)`.
#
# ## Summary
#
# Normalization (resp. un-normalization) can be done by the use of the
# [normalize_vect()][gemseo.algos.design_space.DesignSpace.normalize_vect] method
# (resp. [unnormalize_vect()][gemseo.algos.design_space.DesignSpace.unnormalize_vect]).
#
# A design vector can be retrieved in its normalized form with
# ``design_space.get_current_value(normalize=True)`` when it has a current value.
