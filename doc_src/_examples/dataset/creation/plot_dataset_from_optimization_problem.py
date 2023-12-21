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
Convert a database to a dataset
===============================

In this example,
we will see how to convert a :class:`.Database` to a :class:`.Dataset`.
"""

from __future__ import annotations

from gemseo import execute_algo
from gemseo.problems.analytical.rosenbrock import Rosenbrock

# %%
# Let us solve the :class:`.Rosenbrock` optimization problem
# with the SLSQP algorithm and 10 iterations:
optimization_problem = Rosenbrock()
execute_algo(optimization_problem, "SLSQP", max_iter=10)

# %%
# Then,
# the :class:`.Database` attached to this :class:`.OptimizationProblem`
# can be converted to an :class:`.OptimizationDataset`
# using its method :meth:`~.OptimizationDataset.to_dataset`:
dataset = optimization_problem.to_dataset()
dataset

# %%
# The design variables and output variables are in separate groups.
# You can also use an :class:`.IODataset` instead of an :class:`.OptimizationDataset`:
dataset = optimization_problem.to_dataset(opt_naming=False)
dataset

# %%
# or simply do not separate the variables
dataset = optimization_problem.to_dataset(categorize=False)
dataset
# %%
# .. note::
#     Only design variables and functions (objective function, constraints) are
#     stored in the database. If you want to store state variables, you must add
#     them as observables before the problem is executed. Use the
#     :func:`~.OptimizationProblem.add_observable` method.
