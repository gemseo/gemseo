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
"""# How to cast parameters into different types

## Problem

Design parameters can be defined in different ways:
arrays, dict, rela or complex...

## Solution

The [DesignSpace][gemseo.algos.design_space.DesignSpace] can cast design parameters
into different types.

## Step-by-step guide

You will see how to deal with type casting:

- cast an array to a dict;
- cast a dict into an array;
- cast to complex values;
- convert to integer, if needed.
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
design_space.add_variable(
    "x2", lower_bound=-10, upper_bound=10, type_=design_space.DesignVariableType.INTEGER
)
design_space.add_variable("x3", lower_bound=-10, upper_bound=10)
design_space.add_variable("x4", value=ones(1), lower_bound=-10, upper_bound=10)

# %%
# ### 2. Cast a design point from array to dict
#
# We can cast a design point from `array` to `dict`,
# by means of the
# [convert_array_to_dict()][gemseo.algos.design_space.DesignSpace.convert_array_to_dict] method:
array_point = array([1, 2, 3, 4])
dict_point = design_space.convert_array_to_dict(array_point)
dict_point

# %%
# ### 3. Cast a design point from dict to array
#
# We can cast a design point from `dict` to `array` by means of
# the [convert_dict_to_array()][gemseo.algos.design_space.DesignSpace.convert_dict_to_array] method.
#
# !!! note
#     An optional argument denoted `'variable_names'`,
#     which is a list of string and set at `None` by default,
#     lists all of the variables to consider.
#     If `None`, all design variables are considered.
new_array_point = design_space.convert_dict_to_array(dict_point)
new_array_point

# %%
# ### 4. Cast the current value to complex
#
# We can cast the current value to complex by means of
# the [to_complex()][gemseo.algos.design_space.DesignSpace.to_complex] method:
design_space.set_current_value(array([3.0, 1.0, 1.0, 1.0]))
design_space.to_complex()
design_space.get_current_value()

# %%
# ### 5. Cast the right component values of a vector to integer
#
# For a given vector where some components should be integer,
# it is possible to round them by means of
# the [round_vect()][gemseo.algos.design_space.DesignSpace.round_vect] method:
vector = array([1.3, 3.4, 3.6, -1.4])
rounded_vector = design_space.round_vect(vector)
rounded_vector

# %%
# ## Summary
#
# You can transform a design vector with:
#
# - [convert_array_to_dict()][gemseo.algos.design_space.DesignSpace.convert_array_to_dict] to cast an array into a dict;
# - [convert_dict_to_array()][gemseo.algos.design_space.DesignSpace.convert_dict_to_array] to cast a dict into an array;
# - [to_complex()][gemseo.algos.design_space.DesignSpace.to_complex] to cast into complex;
# - [round_vect()][gemseo.algos.design_space.DesignSpace.round_vect] to take into account integers.
