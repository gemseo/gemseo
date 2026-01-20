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

r"""# Use a Python function

## Problem

How can I create a discipline from a Python function?

## Solution

Instantiate
the [AutoPyDiscipline][gemseo.disciplines.auto_py.AutoPyDiscipline] class
from a function computing the outputs, and possibly a function computing the Jacobian.

## Step-by-step guide
"""

from __future__ import annotations

from numpy import array

from gemseo.disciplines.auto_py import AutoPyDiscipline

# %%
# ### 1. Create the Python function


def f(x, y=0.0):
    """A simple Python function taking float numbers and returning a float number.

    Args:
        x: The first input variable.
        y: The second input variable.

    Returns:
        The Jacobian.
    """
    z = x + 2 * y
    return z


# %%
# !!! warning
#
#     Note that the Python function must return
#     one (e.g. `return z`) or more *variables* (e.g. `return z1, z2`) .
#     The following Python function is not allowed
#     as it returns an *expression* instead of a variable:
#
#     ``` python
#     def g(x, y=0.):
#         """A simple Python function returning an expression."""
#         return x + 2*y
#     ```
#
# !!! note
#
#     Note also that by default,
#     the arguments and the returned variables of the Python function
#     are supposed to be either `float` numbers
#     or NumPy arrays with dimensions greater than 1.
#     At the end of this guide, we will see how to use other types.
#
# ### 2. Create the discipline from this function
discipline = AutoPyDiscipline(f)

# %%
# The input variables of the discipline are the arguments of the Python function `f`:
discipline.io.input_grammar.names

# %%
# the output variables of the discipline are the variables returned by `f`:
discipline.io.output_grammar.names

# %%
# and the default input values of the discipline
# are the default values of the arguments of `f`:
discipline.io.input_grammar.defaults

# %%
# ### 3. Execute the discipline
#
# Using the default input values (note that sole `"y"` as a default input value):
discipline.execute({"x": array([1.0])})

# %%
# or custom ones:
discipline.execute({"x": array([1.0]), "y": array([-3.2])})

# %%
# !!! warning
#
#     You may have noticed that
#     the input data are passed to the [AutoPyDiscipline][gemseo.disciplines.auto_py.AutoPyDiscipline] as NumPy arrays
#     even if the Python function `f` is expecting `float` numbers.
#
# ### 4. Use a Jacobian function


def df(x, y):
    """Function returning the Jacobian of z=f(x,y).

    Args:
        x: The first input variable.
        y: The second input variable.

    Returns:
        The Jacobian, shaped as (output_dimension, input_dimension).
    """
    z = array([[1.0, 2.0]])
    return z


discipline = AutoPyDiscipline(f, py_jac=df)
discipline.linearize(input_data={"x": array([1.0])}, compute_all_jacobians=True)


# %%
# ### 5.  Use custom types
#
# By default,
# the [AutoPyDiscipline][gemseo.disciplines.auto_py.AutoPyDiscipline] assumes that
# the arguments and the returned variables of the Python function are
# either `float` numbers or NumPy arrays with dimensions greater than 1.
# This behaviour can be changed in two different ways.
#
# #### NumPy arrays
#
# We can force [AutoPyDiscipline][gemseo.disciplines.auto_py.AutoPyDiscipline]
# to consider all arguments and variables as NumPy arrays
# by setting the option `use_arrays` to `True`,
# as illustrated here:
def copy_array(a):
    a_copy = a.copy()
    return a_copy


discipline = AutoPyDiscipline(copy_array, use_arrays=True)
discipline.execute({"a": array([1.0])})

# %%
# #### User types
#
# We can also define specific types for each argument and return variable.
#
# !!! warning
#
#     If you forget to annotate an argument or a return variable,
#     all the types you have specified will be ignored.
#
# As a very simple example,
# we can consider a Python function which replicates a string *n* times:


def replicate_string(string: str = "a", n: int = 3) -> str:
    final_string = string * n
    return final_string


discipline = AutoPyDiscipline(replicate_string)

# %%
# Execution with default input values:
discipline.execute()

# %%
# Execution with custom input values:
discipline.execute({"string": "ab", "n": 5})

# %%
# ## Summary
#
# A discipline can be created
# from the [AutoPyDiscipline][gemseo.disciplines.auto_py.AutoPyDiscipline] class
# using Python functions.
#
