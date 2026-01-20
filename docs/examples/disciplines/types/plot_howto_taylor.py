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
r"""# Use a Taylor linearization

## Problem

How can I create a discipline
computing the first-order Taylor polynomial of a given discipline
at a specific point?

## Solution

Instantiate
the [TaylorDiscipline][gemseo.disciplines.taylor.TaylorDiscipline] class
from the given discipline.

## Step-by-step guide
"""

from __future__ import annotations

from numpy import array

from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.disciplines.taylor import TaylorDiscipline

# %%
# Consider the discipline $f(x)=(\sin(x_1)+\cos(x_2),\cos(x_1)+\sin(x_2))$.
#
# ### 1. Create the reference discipline.
discipline = AnalyticDiscipline(
    {"y1": "sin(x1)+cos(x2)", "y2": "cos(x1)+sin(x2)"}, name="f"
)

# %%
# ### 2. Create the Taylor discipline
#
# The first-order Taylor polynomial of $f$ at an input point $a$ is
#
# $$
#    f_a(x) = \begin{pmatrix}
#              \sin(a_1) + \cos(a_2) + \cos(a_1)(x_1-a_1) - \sin(a_2)(x_2-a_2)\\
#              \cos(a_1) + \sin(a_2) - \sin(a_1)(x_1-a_1) + \cos(a_2)(x_2-a_2)
#             \end{pmatrix}.
# $$
#
# #### Taylor at default input values
#
# By default,
# the first-order Taylor polynomial is defined
# at the default input values of the reference discipline.
taylor_at_defaults = TaylorDiscipline(discipline, name="f_defaults")
taylor_at_defaults.execute()

# %%
# #### Taylor at specific input values
#
# Use the `input_data` argument for changing the input point $a$:
taylor_at_specific = TaylorDiscipline(
    discipline, name="f_specific", input_data={"x1": array([0.2]), "x2": array([-0.8])}
)
taylor_at_specific.execute()

# %%
# ## Summary
#
# The [TaylorDiscipline][gemseo.disciplines.taylor.TaylorDiscipline] can be used
# to compute the first-order Taylor polynomial of a reference discipline.
#
# !!! note
#
#     When the reference discipline is almost linear over the input range of interest
#     and provides the analytical derivatives,
#     a [TaylorDiscipline][gemseo.disciplines.taylor.TaylorDiscipline] can be a very relevant surrogate model.
#     Indeed,
#     it can be built with only 1 evaluation
#     whereas a simple linear model would need $1+d$ evaluations
#     where $d$ is the dimension of the input space.
#
