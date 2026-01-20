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

r"""# Use analytic expressions

## Problem

Given simple mathematical expressions,
how can I create a differentiable discipline?

## Solution

Instantiate
the [AnalyticDiscipline][gemseo.disciplines.analytic.AnalyticDiscipline] class
from a dictionary of the form `{output_name: expressions, ...}`.

## Step-by-step guide
"""

from __future__ import annotations

from numpy import array

from gemseo.disciplines.analytic import AnalyticDiscipline

# %%
# Consider the expressions $y_1 = 2x^2$ and $y_2 = 5+3x^2+z^3$.
#
# ### 1. Create the dictionary of expressions
#
# The keys in the dictionary are the names of the outputs,
# and the values are the expressions.
expressions = {"y_1": "2*x**2", "y_2": "5+3*x**2+z**3"}

# %%
# Please refer to the [sympy documentation](https://docs.sympy.org/)
# for the syntax to use in expressions.
#
# ### 2. Create the discipline
discipline = AnalyticDiscipline(expressions, name="f")

# %%
# !!! warning
#
#     This class only supports scalar input and output variables.
#
#
# ### 3. Execute the discipline
#
# #### Default input values
#
# By default, the default input values are equal to 0.
discipline.execute()

# %%
# #### Custom input values
discipline.execute({"x": array([2.0]), "z": array([3.0])})


# %%
# #### Derivatives
#
# The discipline is automatically differentiable;
# you only have to enable the calculation of all derivatives.
discipline.linearize({"x": array([2.0]), "z": array([3.0])}, compute_all_jacobians=True)

# %%
# ## Summary
#
# A discipline can be created
# from a dictionary of the form `{output_name: expressions, ...}`
# using the [AnalyticDiscipline][gemseo.disciplines.analytic.AnalyticDiscipline] class.
# It computes the derivatives automatically using symbolic computation.
#
