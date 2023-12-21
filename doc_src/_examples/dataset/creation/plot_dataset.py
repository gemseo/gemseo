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
Dataset
=======

In this example,
we will see how to build and manipulate a :class:`.Dataset`.

From a conceptual point of view,
a :class:`.Dataset` is a tabular data structure
whose rows are the entries, a.k.a. observations or indices,
and whose columns are the features, a.k.a. quantities of interest.
These features can be grouped by variable identifier
which is a tuple ``(group_name, variable_name)``
and has a dimension equal to the number of components of the variable, a.k.a. dimension.
A feature is a tuple ``(group_name, variable_name, component)``.

From a software point of view,
a :class:`.Dataset` is a particular `pandas DataFrame
<https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`__.
"""

from __future__ import annotations

from numpy import array
from pandas import DataFrame

from gemseo.datasets.dataset import Dataset

# %%
# Instantiation
# -------------
#
# At instantiation,
dataset = Dataset()

# %%
# a dataset has the same name as its class:
dataset.name

# %%
# We can use a more appropriate name at instantiation:
dataset_with_custom_name = Dataset(dataset_name="Measurements")
dataset_with_custom_name.name

# %%
# or change it after instantiation:
dataset_with_custom_name.name = "simulations"
dataset_with_custom_name.name

# %%
# Let us check that the class :class:`.Dataset` derives from ``pandas.DataFrame``:
isinstance(dataset, DataFrame)

# %%
# Add a variable
# --------------
#
# Then,
# we can add data by variable name:
dataset.add_variable("a", array([[1, 2], [3, 4]]))
dataset

# %%
# Note that
# the columns of the dataset use the multi-level index ``(GROUP, VARIABLE, COMPONENT)``.
#
# By default,
# the variable is placed in the group
dataset.DEFAULT_GROUP

# %%
# The attribute ``group_name`` allows to use another group:
dataset.add_variable("b", array([[-1, -2, -3], [-4, -5, -6]]), "inputs")
dataset

# %%
# In the same way,
# for a variable of dimension 2,
# the components are 0 and 1.
# We can use other values with the attribute ``components``:
dataset.add_variable("c", array([[1.5], [3.5]]), components=[3])
dataset

# %%
# Add a group of variables
# ------------------------
#
# Note that the data can also be added by group:
dataset.add_group(
    "G1", array([[-1.1, -2.1, -3.1], [-4.1, -5.1, -6.1]]), ["p", "q"], {"p": 2, "q": 1}
)
dataset

# %%
# The dimensions of the variables ``{"p": 2, "q": 1}`` are not mandatory
# when the number of variable names is equal to the number of columns of the data array:
dataset.add_group("G2", array([[1.1, 2.1, 3.1], [4.1, 5.1, 6.1]]), ["x", "y", "z"])
dataset

# %%
# In the same way,
# the name of the variable is not mandatory;
# when missing,
# ``"x"`` will be considered
# with a dimension equal to the number of columns of the data array:
dataset.add_group("G3", array([[1.2, 2.2], [3.2, 4.2]]))
dataset

# %%
# Convert to a dictionary of arrays
# ---------------------------------
# Sometimes,
# it can be useful to have a dictionary view of the dataset
# with NumPy arrays as values:
dataset.to_dict_of_arrays()

# %%
# We can also flatten this dictionary:
dataset.to_dict_of_arrays(False)


# %%
# Get information
# ---------------
#
# Some properties
# ~~~~~~~~~~~~~~~
#
# At any time,
# we can access to the names of the groups of variables:
dataset.group_names

# %%
# and to the total number of components per group:
dataset.group_names_to_n_components

# %%
# Concerning the variables,
# note that we can use the same variable name in two different groups.
# The (unique) variable names can be accessed with
dataset.variable_names

# %%
# while the total number of components per variable name can be accessed with
dataset.variable_names_to_n_components

# %%
# Lastly,
# the variable identifiers ``(group_name, variable_name)`` can be accessed with
dataset.variable_identifiers

# %%
# Some getters
# ~~~~~~~~~~~~
#
# We can also easily access to the group of a variable:
dataset.get_group_names("x")

# %%
# and to the names of the variables included in a group:
dataset.get_variable_names("G1")

# %%
# The components of a variable located in a group can be accessed with
dataset.get_variable_components("G2", "y")

# %%
# Lastly,
# the columns of the dataset have string representations:
dataset.get_columns()

# %%
# that can be split into tuples:
dataset.get_columns(as_tuple=True)

# %%
# We can also consider a subset of the columns:
dataset.get_columns(["c", "y"])

# %%
# Renaming
# --------
# It is quite easy to rename a group:
dataset.rename_group("G1", "foo")
dataset.group_names

# %%
# or a variable:
dataset.rename_variable("x", "bar", "G2")
dataset.rename_variable("y", "baz")
dataset.variable_names

# %%
# Note that the group name ``"G2"`` allows to rename ``"x"`` only in ``"G2"``;
# without this information,
# the method would have renamed ``"x"`` in both ``"G2"`` and ``"G3"``.

# %%
# Transformation to a variable
# ----------------------------
# One can use a function applying to a NumPy array
# to transform the data associated with a variable,
# for instance a twofold increase:
dataset.transform_data(lambda x: 2 * x, variable_names="bar")

# %%
# Get a view of the dataset
# -------------------------
# The method :meth:`~.Dataset.get_view` returns a view of the dataset
# by using masks built from variable names, group names, components and row indices.
# For instance,
# we can get a view of the variables ``"b"`` and ``"x"``:
dataset.get_view(variable_names=["b", "x"])

# %%
# or a view of the group ``"inputs"``:
dataset.get_view("inputs")

# %%
# We can also combine the keys:
dataset.get_view(variable_names=["b", "x"], components=[0])

# %%
# Update some data
# ----------------
# To complete this example,
# we can update the data
# by using masks built from variable names, group names, components and row indices:
dataset.update_data([[10, 10, 10]], "inputs", indices=[1])
dataset
