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
Dataset from a numpy array
==========================

In this example, we will see how to build a :class:`.Dataset` from an numpy
array. For that, we need to import this :class:`.Dataset` class:
"""
from __future__ import annotations

from gemseo.api import configure_logger
from gemseo.core.dataset import Dataset
from numpy import concatenate
from numpy.random import rand

configure_logger()


##############################################################################
# Synthetic data
# --------------
# Let us consider three parameters:
#
# - x_1 with dimension 1,
# - x_2 with dimension 2,
# - y_1 with dimension 3.
dim_x1 = 1
dim_x2 = 2
dim_y1 = 3
sizes = {"x_1": dim_x1, "x_2": dim_x2, "y_1": dim_y1}
groups = {"x_1": "inputs", "x_2": "inputs", "y_1": "outputs"}

##############################################################################
# We generate 5 random samples of the inputs where:
#
# - x_1 is stored in the first column,
# - x_2 is stored in the 2nd and 3rd columns
#
# and 5 random samples of the outputs.

n_samples = 5
inputs = rand(n_samples, dim_x1 + dim_x2)
inputs_names = ["x_1", "x_2"]
outputs = rand(n_samples, dim_y1)
outputs_names = ["y_1"]
data = concatenate((inputs, outputs), 1)
data_names = inputs_names + outputs_names

##############################################################################
# Create a dataset
# ----------------
# using default names
# ~~~~~~~~~~~~~~~~~~~
# We build a :class:`.Dataset` and initialize from the whole data:

dataset = Dataset(name="random_dataset")
dataset.set_from_array(data)
print(dataset)

##############################################################################
# using particular names
# ~~~~~~~~~~~~~~~~~~~~~~
# We can also use the names of the variables, rather than the default ones
# fixed by the class:
dataset = Dataset(name="random_dataset")
dataset.set_from_array(data, data_names, sizes)
print(dataset)
print(dataset.data)

##############################################################################
# .. warning::
#
#    The number of variables names must be equal to the number of columns of
#    the data array. Otherwise, the user has to specify the sizes of the
#    different variables by means of a dictionary and be careful that the
#    total size is equal to this number of columns.

##############################################################################
# using particular groups
# ~~~~~~~~~~~~~~~~~~~~~~~
# We can also use the notions of groups of variables:
dataset = Dataset(name="random_dataset")
dataset.set_from_array(data, data_names, sizes, groups)
print(dataset)
print(dataset.data)

##############################################################################
# .. note::
#
#    The groups are specified by means of a dictionary where indices are the
#    variables names and values are the groups. If a variable is missing,
#    the default group 'parameters' is considered.

##############################################################################
# storing by names
# ~~~~~~~~~~~~~~~~
# We can also store the data by variables names rather than by groups.
dataset = Dataset(name="random_dataset", by_group=False)
dataset.set_from_array(data, data_names, sizes, groups)
print(dataset)
print(dataset.data)

##############################################################################
# .. note::
#
#    The choice to be made between a storage by group and a storage by
#    variables names aims to limit the number of memory copies of numpy arrays.
#    It mainly depends on how the dataset is used and for what purposes.
#    For example, if we want to build a machine learning algorithm from both
#    input and output data, we only have to access the data by group and in
#    this case, storing the data by group is recommended. Conversely, if we
#    want to use the dataset for post-processing purposes, by accessing the
#    variables of the dataset from their names, the storage by variables names
#    is preferable.

##############################################################################
# Access properties
# -----------------
# Variables names
# ~~~~~~~~~~~~~~~
# We can access the variables names:
print(dataset.variables)

##############################################################################
# Variables sizes
# ~~~~~~~~~~~~~~~
# We can access the variables sizes:
print(dataset.sizes)

##############################################################################
# Variables groups
# ~~~~~~~~~~~~~~~~
# We can access the variables groups:
print(dataset.groups)

##############################################################################
# Access data
# -----------
# Access by group
# ~~~~~~~~~~~~~~~
# We can get the data by group, either as an array (default option):
print(dataset.get_data_by_group("inputs"))
##############################################################################
# or as a dictionary indexed by the variables names:
print(dataset.get_data_by_group("inputs", True))

##############################################################################
# Access by variable name
# ~~~~~~~~~~~~~~~~~~~~~~~
# We can get the data by variables names,
# either as a dictionary indexed by the variables names (default option):
print(dataset.get_data_by_names(["x_1", "y_1"]))
##############################################################################
# or as an array:
print(dataset.get_data_by_names(["x_1", "y_1"], False))

##############################################################################
# Access all data
# ~~~~~~~~~~~~~~~
# We can get all the data, either as a large array:
print(dataset.get_all_data())
##############################################################################
# or as a dictionary indexed by variables names:
print(dataset.get_all_data(as_dict=True))
##############################################################################
# We can get these data sorted by category, either with a large array for each
# category:
print(dataset.get_all_data(by_group=False))
##############################################################################
# or with a dictionary of variables names:
print(dataset.get_all_data(by_group=False, as_dict=True))
