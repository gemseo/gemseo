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
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Create a discipline from a Python function
==========================================

There is a simplified and straightforward way of integrating a discipline
from a Python function that:

- returns variables,
  e.g. ``return x`` or ``return x, y``,
  but no expression like ``return a+b`` or ``return a+b, y``,
- must have a default value per argument
  if the :class:`.AutoPyDiscipline` is used by an ``MDA``
  (deriving from :class:`.BaseMDA`),
  as in the case of :class:`.MDF` and :class:`.BiLevel` formulations.
"""

from __future__ import annotations

from numpy import array

from gemseo import configure_logger
from gemseo import create_discipline

configure_logger()
# %%
# In this example,
# we will illustrate this feature
# using the Python function:


def f(x, y=0.0):
    """A simple Python function taking float numbers and returning a float number."""
    z = x + 2 * y
    return z


# %%
# .. warning::
#    Note that the Python function must return one or more *variables*.
#    The following Python function would not be suitable
#    as it returns an *expression* instead of a variable:
#
#    .. code::
#
#        def g(x, y=0.):
#            """A simple Python function returning an expression."""
#            return x + 2*y
#
# .. note::
#    Note also that by default,
#    the arguments and the returned variables of the Python function
#    are supposed to be either ``float`` numbers
#    or NumPy arrays with dimensions greater than 1.
#    At the end of the example, we will see how to use other types.
#
# Then, we can consider the
# :class:`.AutoPyDiscipline` class
# to wrap this Python function into a :class:`.Discipline`.
# For that,
# we can use the :func:`.create_discipline` high-level function
# with the string ``"AutoPyDiscipline"`` as first argument:
discipline = create_discipline("AutoPyDiscipline", py_func=f)

# %%
# The input variables of the discipline are the arguments of the Python function ``f``:
discipline.io.input_grammar.names
# %%
# the output variables of the discipline are the variables returned by ``f``:
discipline.io.output_grammar.names
# %%
# and the default input values of the discipline
# are the default values of the arguments of ``f``:
discipline.io.input_grammar.defaults

# %%
# .. note::
#    The argument ``x`` of the Python function ``f`` shall have a default value
#    when the discipline is used by an ``MDA`` (deriving from :class:`.BaseMDA`),
#    as in the case of :class:`.MDF` and :class:`.BiLevel` formulations,
#    in presence of strong couplings.
#    This is not the case in this example.

# %%
# Execute the discipline
# ----------------------
# Then,
# we can execute this discipline easily,
# either considering default input values:
discipline.execute({"x": array([1.0])})

# %%
# or custom ones:
discipline.execute({"x": array([1.0]), "y": array([-3.2])})

# %%
# .. warning::
#    You may have noticed that
#    the input data are passed to the :class:`.AutoPyDiscipline` as NumPy arrays
#    even if the Python function ``f`` is expecting ``float`` numbers.
#
# Define the Jacobian function
# ----------------------------
# Here is an example of a Python function
# returning the Jacobian matrix as a 2D NumPy array:


def df(x, y):
    """Function returning the Jacobian of z=f(x,y)"""
    return array([[1.0, 2.0]])


# %%
# We can create a new :class:`AutoPyDiscipline` from ``f`` and ``df``:
discipline = create_discipline("AutoPyDiscipline", py_func=f, py_jac=df)
# %%
# and compute its Jacobian at ``{"x": array([1.0]), "y": array([0.0])}``:
discipline.linearize(input_data={"x": array([1.0])}, compute_all_jacobians=True)
discipline.jac


# %%
# Use custom types
# ----------------
# By default,
# the :class:`.AutoPyDiscipline` assumes that
# the arguments and the returned variables of the Python function are
# either ``float`` numbers or NumPy arrays with dimensions greater than 1.
# This behaviour can be changed in two different ways.
#
# NumPy arrays
# ~~~~~~~~~~~~
# We can force :class:`.AutoPyDiscipline`
# to consider all arguments and variables as NumPy arrays
# by setting the option ``use_arrays`` to ``True``,
# as illustrated here:
def copy_array(a):
    a_copy = a.copy()
    return a_copy


discipline = create_discipline("AutoPyDiscipline", py_func=copy_array, use_arrays=True)
discipline.execute({"a": array([1.0])})

# %%
# User types
# ~~~~~~~~~~
# We can also define specific types for each argument and return variable.
#
# .. warning::
#    If you forget to annotate an argument or a return variable,
#    all the types you have specified will be ignored.
#
# As a very simple example,
# we can consider a Python function which replicates a string *n* times:


def replicate_string(string: str = "a", n: int = 3) -> str:
    final_string = string * n
    return final_string


# %%
# Then,
# we create the discipline:
discipline = create_discipline("AutoPyDiscipline", py_func=replicate_string)
# %%
# execute it with its default input values:
discipline.execute()
# %%
# and with custom ones:
discipline.execute({"string": "ab", "n": 5})
