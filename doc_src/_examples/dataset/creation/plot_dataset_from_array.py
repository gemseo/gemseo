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
Dataset from a NumPy array
==========================

In this example, we will see how to build a :class:`.Dataset` from an NumPy array.
"""

from __future__ import annotations

from numpy import concatenate
from numpy.random import default_rng

from gemseo import configure_logger
from gemseo.datasets.dataset import Dataset

configure_logger()

rng = default_rng(1)

# %%
# Let us consider three parameters :math:`x_1`, :math:`x_2` and :math:`x_3`
# of size 1, 2 and 3 respectively.
# We generate 5 random samples of the inputs where
#
# - x_1 is stored in the first column,
# - x_2 is stored in the 2nd and 3rd columns
#
# and 5 random samples of the outputs:
n_samples = 5
inputs = rng.random((n_samples, 3))
outputs = rng.random((n_samples, 3))
data = concatenate((inputs, outputs), 1)

# %%
# A dataset with default names
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We create a :class:`.Dataset` from the NumPy array only
# and let GEMSEO give default names to its columns:
dataset = Dataset.from_array(data)
dataset

# %%
# A dataset with custom names
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We can also pass the names and sizes of the variables:
names_to_sizes = {"x_1": 1, "x_2": 2, "y_1": 3}
dataset = Dataset.from_array(data, ["x_1", "x_2", "y_1"], names_to_sizes)
dataset

# %%
# .. warning::
#
#    The number of variables names must be equal to the number of columns of
#    the data array. Otherwise, the user has to specify the sizes of the
#    different variables by means of a dictionary and be careful that the
#    total size is equal to this number of columns.

# %%
# A dataset with custom groups
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We can also use the notions of groups of variables:
groups = {"x_1": "inputs", "x_2": "inputs", "y_1": "outputs"}
dataset = Dataset.from_array(data, ["x_1", "x_2", "y_1"], names_to_sizes, groups)
dataset

# %%
# .. note::
#
#    The groups are specified by means of a dictionary
#    where indices are the variables names and values are the groups.
#    If a variable is missing,
#    the default group :attr:`.Dataset.DEFAULT_GROUP` is considered.
