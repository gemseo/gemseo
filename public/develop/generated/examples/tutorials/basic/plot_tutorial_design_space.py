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
"""# Tutorial - The design space

## Goal

In this tutorial, you will learn to create and manipulate a design space.

As specified in the [user guide][concept-design-space],
a design space defines the optimization variables:
their names, types, dimensions, bounds, and current values (if any).
We will first create an instance of a
[DesignSpace][gemseo.algos.design_space.DesignSpace] with multiple variables,
then learn how to query information about these variables,
and finally modify their values and bounds.
"""

from __future__ import annotations

from numpy import array
from numpy import ones

from gemseo.algos.design_space import DesignSpace

# %%
# ## Step 1 - Create a design space
#
# Let's imagine that we want to build a design space with the following requirements:
#
# - *x1* is a one-dimensional unbounded float variable,
# - *x2* is a one-dimensional unbounded integer variable,
# - *x3* is a two-dimensional unbounded float variable,
# - *x4* is a one-dimensional float variable with lower bound equal to 1,
# - *x5* is a one-dimensional float variable with upper bound equal to 1,
# - *x6* is a one-dimensional unbounded float variable,
# - *x7* is a two-dimensional bounded integer variable with lower bound equal to -1, upper bound equal to 1 and current values to (0,1).
#
# We can create this design space from scratch.
# Use the [add_variable()][gemseo.algos.design_space.DesignSpace.add_variable] method
# of the [DesignSpace][gemseo.algos.design_space.DesignSpace] class to add variables:

design_space = DesignSpace()
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
design_space

# %%
# !!! tip
#     A design space can also be loaded from a file,
#     as shown in [How to import and export a design space from disk](../../howtos/design_space/plot_howto_load_and_save_design_space.md).
#
#
# !!! note
#     For a variable whose `size` is greater than 1,
#     each dimension of this variable is printed (e.g. `x3` and `x7`).
#
# !!! note
#     We can get a list of the variable names with their indices
#     by means of the [get_indexed_variable_names()][gemseo.algos.design_space.DesignSpace.get_indexed_variable_names] method:
#
#     ``` python
#     indexed_variable_names = design_space.get_indexed_variable_names()
#     print(indexed_variable_names)
#     > ['x1', 'x2', 'x3!0', 'x3!1', 'x4', 'x5', 'x6', 'x7!0', 'x7!1']
#     ```
#
#     We see that the multidimensional variables have an index (here `0` and `1`) preceded by a `!` separator.

# %%
# ## Step 2 - Get information about the design space
#
# The design space information can be retrieved by using different methods.
#
# ### Individual variable information
#
# We can query the size, type, and bounds of a specific variable:

print(f"The size of the `x3` variable is: {design_space.get_size('x3')}")
print(f"The type of the `x2` variable is: {design_space.get_type('x2')}")
print(
    f"Variable `x3` is in [{design_space.get_lower_bound('x3')}, {design_space.get_upper_bound('x3')}]"
)

# %%
# ### Information for multiple variables
#
# Methods also exist for a set of variables:

print(
    f"The lower bounds of variables `x3` and `x4` are: {design_space.get_lower_bounds(['x3', 'x4'])}"
)

# %%
# ### Current values
#
# The current values can be retrieved as an array or as a dictionary:

print(
    f"The current value of the design parameters is: {design_space.get_current_value()}"
)
print(f"Or expressed as a dictionary: {design_space.get_current_value(as_dict=True)}")
print(f"Every variable has a current value: {design_space.has_current_value}")

# %%
# !!! note
#     The result returned by [has_current_value][gemseo.algos.design_space.DesignSpace.has_current_value]
#     is `False` as long as at least one component of one variable is `None`.
#
# !!! warning
#     The current value can only be retrieved when
#     [has_current_value][gemseo.algos.design_space.DesignSpace.has_current_value]
#     returns `True`.
#
# ### Check the active bounds
#
# We can get the active bounds by means of the [get_active_bounds()][gemseo.algos.design_space.DesignSpace.get_active_bounds] method,
# either at current parameter values or at a given point.
#
# At the current design value:

active_at_current_x = design_space.get_active_bounds()
active_at_current_x

# %%
# At a given point:

active_at_given_point = design_space.get_active_bounds(
    array([1.0, 10, 0, 0, 0, 0, 0, 1.0, 1.0])
)
active_at_given_point

# %%
# ## Step 3 - Modify the design space
#
# We can change the current values and bounds contained in a design space.
#
# ### Change all current values at once
#
# We can set all current values to a given array:

design_space.set_current_value(ones(9))
design_space

# %%
# ### Change a specific variable
#
# We can also change the current value of a specific variable:

design_space.set_current_variable("x1", array([3.0]))
design_space

# %%
# ### Update bounds
#
# We can modify the lower and upper bounds of variables:

design_space.set_lower_bound("x1", array([-10.0]))
design_space.set_lower_bound("x2", array([-10]))
design_space.set_lower_bound("x3", array([-10.0, -10.0]))
design_space.set_lower_bound("x6", array([-10.0]))

design_space.set_upper_bound("x1", array([10.0]))
design_space.set_upper_bound("x2", array([10]))
design_space.set_upper_bound("x3", array([10.0, 10.0]))
design_space.set_upper_bound("x6", array([10.0]))

design_space

# %%
# ## Key takeaways
#
# In this tutorial, you've learned to:
#
# - create a design space from scratch by means of the [create_design_space()][gemseo.create_design_space] high-level function;
# - add variables with the [add_variable()][gemseo.algos.design_space.DesignSpace.add_variable] method;
# - get information about a given design space by using different getters / attributes;
# - modify the value and the boundaries of design variables.
#
# ## How-to guides
#
# For further information,
# please refer to the following how-to guides:
#
# - [How to import and export a design space from disk](../../howtos/design_space/plot_howto_load_and_save_design_space.md)
# - [How to reduce a design space](../../howtos/design_space/plot_howto_reduce_a_design_space.md)
# - [How to project parameters into boundaries](../../howtos/design_space/plot_howto_project_into_bounds.md)
# - [How to (un)normalize design parameters](../../howtos/design_space/plot_howto_normalize_design_vector.md)
# - [How to cast parameters into different types](../../howtos/design_space/plot_howto_cast_design_vector.md)
