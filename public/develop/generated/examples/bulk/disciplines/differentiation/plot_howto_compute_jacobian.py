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
# Compute the Jacobian of a discipline

## Problem

You want to compute the Jacobian of your discipline.

## Solution

You can compute the derivatives of some outputs of a
[Discipline][gemseo.core.discipline.discipline.Discipline]
with respect to some inputs with the method
[linearize()][gemseo.core.discipline.discipline.Discipline.linearize].

## Step-by-step guide
"""

from __future__ import annotations

from numpy import array

from gemseo.disciplines.analytic import AnalyticDiscipline

# %%
# ### 1. Create a discipline
discipline = AnalyticDiscipline({"y": "a**2+b", "z": "a**3+b**2"})

# %%
# ### 2. Define the differentiated inputs and ouputs
#
# You need to set the input variables
# with respect to which to compute the Jacobian of the output ones.
# For that,
# use the method [add_differentiated_inputs()][gemseo.core.discipline.discipline.Discipline.add_differentiated_inputs].
# You also need to set these output variables:
# with the method [add_differentiated_outputs()][gemseo.core.discipline.discipline.Discipline.add_differentiated_outputs].
# For instance,
# you may want to only compute the derivative of `"z"` with respect to `"a"`:
discipline.add_differentiated_inputs(["a"])
discipline.add_differentiated_outputs(["z"])

# %%
# !!! note
#     Non-numeric variables (e.g. string arrays) are automatically filtered out
#     when no explicit `input_names` or `output_names` are given.
#     This filtering is only supported with
#     [JSONGrammar][gemseo.core.grammars.json_grammar.JSONGrammar] and
#     [PydanticGrammar][gemseo.core.grammars.pydantic_grammar.PydanticGrammar].

# %%
# ### 3. Compute the derivatives
#
# Use the method
# [linearize()][gemseo.core.discipline.discipline.Discipline.linearize]
# to compute the derivatives:
jacobian_data = discipline.linearize()
jacobian_data

# %%
# By default,
# GEMSEO uses [Discipline.default_input_data][gemseo.core.discipline.discipline.Discipline.default_input_data] as input data
# for which to compute the Jacobian on.
# `input_data` can be changed:
jacobian_data = discipline.linearize({"a": array([1.0])})
jacobian_data

# %%
# We can also force the discipline to compute
# the derivatives of all the outputs with respect to all the inputs.
jacobian_data = discipline.linearize(compute_all_jacobians=True)
jacobian_data

# %%
# ## Summary
#
# First, you need to set the input variables
# with respect to which to compute the Jacobian of the output ones, with the
# [add_differentiated_inputs()][gemseo.core.discipline.discipline.Discipline.add_differentiated_inputs]
# and
# [add_differentiated_outputs()][gemseo.core.discipline.discipline.Discipline.add_differentiated_outputs]
# methods.
# Then, you can compute the derivatives of your discipline with the
# [linearize()][gemseo.core.discipline.discipline.Discipline.linearize] method.
#
# ## One step further
#
# You can learn how to change the way derivatives are computed,
# by following [this HOW-TO][change-the-differentiation-settings].
