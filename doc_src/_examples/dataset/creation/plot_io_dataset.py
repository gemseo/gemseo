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
The input-output dataset
========================

The :class:`.IODataset` proposes two particular group names,
namely :attr:`~.IODataset.INPUT_GROUP` and :attr:`~.IODataset.OUTPUT_GROUP`.
This particular :class:`.Dataset` is useful
for supervised machine learning and sensitivity analysis.
"""

from __future__ import annotations

from gemseo.datasets.io_dataset import IODataset

# %%
# First,
# we instantiate the :class:`.IODataset`:
dataset = IODataset()

# %%
# and add some input and output variables
# using the methods
# :meth:`~.IODataset.add_input_variable` and :meth:`~.IODataset.add_output_variable`
# that are based on :meth:`.Dataset.add_variable`:
dataset.add_input_variable("a", [[1.0, 2.0], [4.0, 5.0]])
dataset.add_input_variable("b", [[3.0], [6.0]])
dataset.add_output_variable("c", [[-1.0], [-2.0]])
# %%
# as well as another variable:
dataset.add_variable("x", [[10.0], [20.0]])
dataset

# %%
# We could also do the same with the methods
# :meth:`~.IODataset.add_input_group` and :meth:`~.IODataset.add_output_group`.
dataset = IODataset()
dataset.add_input_group(
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], ["a", "b"], {"a": 2, "b": 1}
)
dataset.add_output_group([[-1.0], [-2.0]], ["c"])
dataset.add_variable("x", [[10.0], [20.0]])
dataset

# %%
# Then,
# we can easily access the names of the input and output variables:
dataset.input_names, dataset.output_names
# %%
# and those of all variables:
dataset.variable_names

# %%
# The :class:`.IODataset` provides also the number of samples:
dataset.n_samples
# %%
# and the samples:
dataset.samples

# %%
# Lastly,
# we can get the input data as an :class:`.IODataset` view:
dataset.input_dataset

# %%
# and the same for the output data:
dataset.output_dataset
