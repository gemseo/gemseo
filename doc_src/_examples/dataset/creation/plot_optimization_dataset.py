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
The optimisation dataset
========================

The :class:`.OptimizationDataset` proposes several particular group names,
namely :attr:`~.IODataset.DESIGN_GROUP`,
:attr:`~.IODataset.OBJECTIVE_GROUP`,
:attr:`~.IODataset.OBSERVABLE_GROUP`,
and :attr:`~.IODataset.CONSTRAINT_GROUP`.
This particular :class:`.Dataset` is useful
to post-process an optimization history.
"""

from __future__ import annotations

from gemseo.datasets.optimization_dataset import OptimizationDataset

# %%
# First,
# we instantiate the :class:`.OptimizationDataset`:
dataset = OptimizationDataset()

# %%
# and add some data of interest
# using the methods
# :meth:`~.OptimizationDataset.add_design_variable`,
# :meth:`~.OptimizationDataset.add_constraint_variable`,
# :meth:`~.OptimizationDataset.add_objective_variable`,
# and :meth:`~.OptimizationDataset.add_observable_variable`
# that are based on :meth:`.Dataset.add_variable`:
dataset.add_design_variable("x", [[1.0, 2.0], [4.0, 5.0]])
dataset.add_design_variable("z", [[3.0], [6.0]])
dataset.add_objective_variable("f", [[-1.0], [-2.0]])
dataset.add_constraint_variable("c", [[-0.5], [0.1]])
dataset.add_observable_variable("o", [[-3.0], [8.0]])
# %%
# as well as another variable:
dataset.add_variable("a", [[10.0], [20.0]])
dataset

# %%
# We could also do the same with the methods
# :meth:`~.OptimizationDataset.add_design_group`,
# :meth:`~.OptimizationDataset.add_constraint_group`,
# :meth:`~.OptimizationDataset.add_objective_group`,
# and :meth:`~.OptimizationDataset.add_observable_group`
dataset = OptimizationDataset()
dataset.add_design_group(
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], ["x", "y"], {"x": 2, "y": 1}
)
dataset.add_objective_group([[-1.0], [-2.0]], ["f"])
dataset.add_constraint_group([[-0.5], [0.1]], ["c"])
dataset.add_observable_group([[-3.0], [8.0]], ["o"])
dataset.add_variable("a", [[10.0], [20.0]])
dataset

# %%
# Then,
# we can easily access the names of the different input variables
dataset.design_variable_names
# %%
# the names of the output variables
dataset.constraint_names, dataset.objective_names, dataset.observable_names
# %%
# and the names of all variables:
dataset.variable_names

# %%
# The :class:`.OptimizationDataset` provides also the number of iterations:
dataset.n_iterations
# %%
# and the iterations:
dataset.iterations

# %%
# Lastly,
# we can get the design data as an :class:`.OptimizationDataset` view:
dataset.design_dataset

# %%
# and the same for the other data groups,
dataset.constraint_dataset
# %%
dataset.objective_dataset
# %%
dataset.observable_dataset
