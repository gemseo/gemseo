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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Charlie Vanaret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""# Chain disciplines

## Problem

You have several disciplines, and you want to execute them sequentially and forward the outputs of each discipline to the subsequent disciplines.

## Solution

GEMSEO can chain the disciplines to execute them sequentially with the
[MDOChain][gemseo.core.chains.chain.MDOChain] discipline.

## Step-by-step guide
"""

from __future__ import annotations

from numpy import array

from gemseo.core.chains.chain import MDOChain
from gemseo.disciplines.analytic import AnalyticDiscipline

# %%
# ### 1. Create your disciplines
discipline_a = AnalyticDiscipline({"y": "2*x"}, name="A")
discipline_b = AnalyticDiscipline({"z": "y+1"}, name="B")

# %%
# ### 2. Chain the disciplines
chain = MDOChain([discipline_a, discipline_b])
chain

# %%
#
# The inputs of the chain are the inputs of the disciplines
# minus the couplings between these disciplines.
# They have the same default values, if any.
# The outputs of the chain are the output of the disciplines.
#
# ### 3. Execute the chain
#
# Executing this chain is equivalent to executing discipline `A`,
# getting its outputs,
# and giving them to discipline `B`.
#
# !!! warning
#     The order of the disciplines matters.
#     `MDOChain([discipline_a, discipline_b])` executes `discipline_a` before `discipline_b`
#     while `MDOChain([discipline_b, discipline_a])` executes `discipline_b` before `discipline_a`.
#
# We can execute the chain using its default inputs:
chain.execute()

# %%
# or by giving inputs:
chain.execute({"x": array([5])})

# %%
# !!! warning
#     The `MDOChain` does not solve couplings,
#     whether they are strong or weak.
#     Regardless of their coupling structure,
#     the disciplines are simply executed sequentially.
#
# ## Summary
#
# Multiple disciplines can be chained together thanks to the
# [MDOChain][gemseo.core.chains.chain.MDOChain].
# The order of execution is specified by the user.
#
# ## One step further
#
# To consider strong couplings,
# please refer to [this how-to guide][manage-nested-coupling-systems].
