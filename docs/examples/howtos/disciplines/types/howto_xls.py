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

r"""# Use an Excel file

## Problem

How can I create a discipline
from an Excel file?

## Solution

Instantiate
the [XLSDiscipline][gemseo.disciplines.wrappers.xls_discipline.XLSDiscipline] class
from the given Excel file.

## Step-by-step guide
"""

from __future__ import annotations

from numpy import array

from gemseo.disciplines.wrappers.xls_discipline import XLSDiscipline

# %%
# ### 1. Create the Excel file
#
# Create the Excel file that will compute the outputs (`c`) from the
# inputs (`a`, `b`). Inputs must be specified in the "Inputs" sheet:
#
# |     | A   | B   |
# |-----|-----|-----|
# | 1   | a   | 3   |
# | 2   | b   | 5   |
#
# "Inputs" sheet setting $a=3$ and $b=5$.
#
# !!! warning
#     The number of rows is arbitrary,
#     but they must be contiguous (no empty lines)
#     and start at line 1.
#
# The same applies for the "Outputs" sheet:
#
# |     | A   | B   |
# |-----|-----|-----|
# | 1   | c   | 8   |
#
# "Outputs" sheet setting $c=8$.
#
# The Excel must get a macro command to compute the outputs according to the inputs.
# By default, this command is called `execute`,
# but can be changed through the `macro_name` argument.
#
# ### 2. Create the discipline
#
# For this basic implementation we only
# need to provide the path to the Excel file:
#
xls_discipline = XLSDiscipline("my_book.xlsx")
xls_discipline.execute({"a": array([1.0])})
c = xls_discipline.io.data["c"]

# %%
# ## Summary
#
# The [XLSDiscipline][gemseo.disciplines.wrappers.xls_discipline.XLSDiscipline]
# can be used to create a discipline from a formatted Excel file.
#
# ## One step further
#
# ### Parallel execution considerations
#
# GEMSEO relies on the [xlswings library](https://www.xlwings.org) to
# communicate with Excel. This imposes some constraints to our
# development. In particular, [we cannot pass xlwings objects between
# processes or
# threads](https://docs.xlwings.org/en/stable/threading_and_multiprocessing.html).
# We have different strategies to comply with this requirement in parallel
# execution, depending on whether we are using multiprocessing,
# multithreading or both.
#
# #### Multiprocessing
#
# In multiprocessing, we recreate the `xlwings` object in each subprocess
# through `__setstate__`. However, the same Excel file cannot be used by
# all the subprocesses at the same time. Which means that we need a unique
# copy of the original file for each one.
#
# The option `copy_xls_at_setstate` shall be set to `True` whenever an
# [XLSDiscipline][gemseo.disciplines.wrappers.xls_discipline.XLSDiscipline] will be used
# in a [DiscParallelExecution][gemseo.core.parallel_execution.discipline_execution.DiscParallelExecution] instance implementing multiprocessing.
#
# #### Multithreading
#
# In multithreading, we recreate the `xlwings` object at each call to the
# [XLSDiscipline][gemseo.disciplines.wrappers.xls_discipline.XLSDiscipline]. Thus, when instantiating an [XLSDiscipline][gemseo.disciplines.wrappers.xls_discipline.XLSDiscipline] that will
# be executed in multithreading, the user must set
# `recreate_book_at_run=True`.
#
# !!! warning
#     An [MDAJacobi][gemseo.mda.jacobi.MDAJacobi] uses multithreading to accelerate its convergence,
#     even if the overall scenario is being run in serial mode. If your
#     [XLSDiscipline][gemseo.disciplines.wrappers.xls_discipline.XLSDiscipline] is inside an [MDAJacobi][gemseo.mda.jacobi.MDAJacobi], you must instantiate it
#     with `recreate_book_at_run=True`.
#
# #### Multiprocessing & Multithreading
#
# There is one last case to consider, which occurs when the
# [XLSDiscipline][gemseo.disciplines.wrappers.xls_discipline.XLSDiscipline]will run in multithreading mode from a subprocess that
# was itself created by a multiprocessing instance. A good example of this
# particular situation is when an [MDOScenario][gemseo.scenarios.mdo.MDOScenario] runs in parallel with an
# [MDAJacobi][gemseo.mda.jacobi.MDAJacobi] that solves the couplings for each sample.
#
# It will be necessary to set both `copy_xls_at_setstate=True` and
# `recreate_book_at_run=True`.
