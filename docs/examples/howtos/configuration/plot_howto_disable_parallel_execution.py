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
"""# Disable the parallel execution

## Problem

The Jacobi MDA uses several threads or processes by default.
You want it to use a single one.

## Solution

Set the option `enable_parallel_execution` of the
[global configuration][concept-global-configuration] to `False`.

## Step-by-step guide
"""

from __future__ import annotations

from gemseo import configuration
from gemseo.discipline import AnalyticDiscipline
from gemseo.mda import MDAJacobi

# %%
# ### Prerequisites
#
# Two coupled disciplines:
disciplines = [
    AnalyticDiscipline({"y": "x/2+1"}),
    AnalyticDiscipline({"x": "y/2+1"}),
]

# %%
# By default,
# the number of threads or processes of a Jacobi MDA
# is the number of CPUs of the machine:
mda = MDAJacobi(disciplines)
mda.settings.n_processes

# %%
# ### 1. Disable the parallel execution
configuration.enable_parallel_execution = False

# %%
# ### 2. Check the number of processes
#
# An MDA created after the change uses a single thread or process:
MDAJacobi(disciplines).settings.n_processes

# %%
# while an MDA created before the change keeps its number of processes:
mda.settings.n_processes

# %%
# ## Summary
#
# Set `configuration.enable_parallel_execution` to `False`
# before creating the objects.
# The default number of threads or processes is then 1
# for the parallel MDAs,
# [ParallelDisciplineChain][gemseo.discipline.chain.parallel_chain.ParallelDisciplineChain]
# and [CallableParallelExecution][gemseo.core.parallel_execution.callable_parallel_execution.CallableParallelExecution].
