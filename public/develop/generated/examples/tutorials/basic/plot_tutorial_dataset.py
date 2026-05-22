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
"""# Tutorial - Manipulating a dataset

## Goal

This tutorial introduces the dataset classes available in GEMSEO
for storing and manipulating tabular data:

- [Dataset][gemseo.datasets.dataset.Dataset] — the generic base class,
- [IODataset][gemseo.datasets.io_dataset.IODataset] — a specialization for
  input/output data (supervised machine learning, sensitivity analysis),
- [OptimizationDataset][gemseo.datasets.optimization_dataset.OptimizationDataset]
  — a specialization for optimization histories.

You will learn how to:

- **create** a dataset and populate it with variables or groups of variables,
- **inspect** its structure using properties and getters,
- **rename** groups and variables,
- **transform** data in place,
- **query** the dataset with `get_view()`,
- **update** entries selectively,
- **export** the dataset to a dictionary of NumPy arrays,
- **use** the [IODataset][gemseo.datasets.io_dataset.IODataset] and [OptimizationDataset][gemseo.datasets.optimization_dataset.OptimizationDataset] subclasses
  and their domain-specific helpers.

A [Dataset][gemseo.datasets.dataset.Dataset] is a special
[pandas MultiIndex DataFrame](https://pandas.pydata.org/docs/user_guide/advanced.html).
Its columns follow the three-level index `(GROUP, VARIABLE, COMPONENT)`,
where a *feature* is a `(group_name, variable_name, component)` triplet
and a *variable identifier* is the unique pair `(group_name, variable_name)`.
Both subclasses inherit this structure and simply pre-define
a set of meaningful group names.

!!! note
    A GEMSEO process can export its execution data directly into a dataset.
    In most cases, you will not need to create one manually.
"""

from __future__ import annotations

from numpy import array
from pandas import DataFrame

from gemseo.datasets.dataset import Dataset
from gemseo.datasets.io_dataset import IODataset
from gemseo.datasets.optimization_dataset import OptimizationDataset

# %%
# ## Step 1 — Create a dataset and add variables
#
# Instantiate an empty dataset.
# By default its name equals the class name `"Dataset"`.
# Pass `dataset_name` to give it a meaningful label.
dataset = Dataset(dataset_name="MyDataset")
dataset.name

# %%
# A [Dataset][gemseo.datasets.dataset.Dataset] is a genuine `pandas.DataFrame`:
isinstance(dataset, DataFrame)

# %%
# Use [add_variable()][gemseo.datasets.dataset.Dataset.add_variable]
# to populate the dataset one variable at a time.
# The data must be a 2-D NumPy array shaped as `(n_entries, n_components)`.
# When no group is specified,
# the variable is placed in the default group
# (see [DEFAULT_GROUP][gemseo.datasets.dataset.Dataset.DEFAULT_GROUP]).
dataset.add_variable("a", array([[1, 2], [3, 4]]))
dataset

# %%
# The default group name is:
dataset.DEFAULT_GROUP

# %%
# To assign a variable to a specific group, pass the `group_name` argument:
dataset.add_variable("b", array([[-1, -2, -3], [-4, -5, -6]]), "inputs")
dataset

# %%
# By default the component indices start at `0`.
# Use `components` to assign custom indices.
# Here we place the single component of `"c"` at index `3`:
dataset.add_variable("c", array([[1.5], [3.5]]), components=[3])
dataset

# %%
# ## Step 2 — Add a whole group of variables at once
#
# [add_group()][gemseo.datasets.dataset.Dataset.add_group] lets you load
# a block of data together with the variable names and their dimensions.
# The dimensions dictionary `{"p": 2, "q": 1}` states that
# `"p"` occupies 2 columns and `"q"` occupies 1 column:
dataset.add_group(
    "G1",
    array([[-1.1, -2.1, -3.1], [-4.1, -5.1, -6.1]]),
    ["p", "q"],
    {"p": 2, "q": 1},
)
dataset

# %%
# The dimensions dictionary is optional when the number of variable names
# equals the number of columns of the data array
# (each variable then has exactly one component):
dataset.add_group("G2", array([[1.1, 2.1, 3.1], [4.1, 5.1, 6.1]]), ["x", "y", "z"])
dataset

# %%
# Both `variable_names` and `variable_name_to_n_components` are optional.
# When omitted, GEMSEO uses the
# [DEFAULT_VARIABLE_NAME][gemseo.datasets.dataset.Dataset.DEFAULT_VARIABLE_NAME]
# `"x"` with as many components as there are columns:
dataset.add_group("G3", array([[1.2, 2.2], [3.2, 4.2]]))
dataset

# %%
# ## Step 3 — Inspect the dataset structure
#
# The names of all groups, in alphabetical order:
dataset.group_names

# %%
# The total number of components per group:
dataset.group_name_to_n_components

# %%
# The unique variable names (across all groups), in alphabetical order.
# Note that the same name can appear in several groups;
# only the pair `(group_name, variable_name)` — called a *variable identifier* — is unique:
dataset.variable_names

# %%
# The total number of components per variable name:
dataset.variable_name_to_n_components

# %%
# All variable identifiers as `(group_name, variable_name)` pairs:
dataset.variable_identifiers

# %%
# Find which group(s) contain a given variable:
dataset.get_group_names("x")

# %%
# List the variables inside a given group:
dataset.get_variable_names("G1")

# %%
# Retrieve the component indices of a variable in a given group:
dataset.get_variable_components("G2", "y")

# %%
# Columns can be displayed as human-readable strings ...
dataset.get_columns()

# %%
# ... or as `(group_name, variable_name, component)` tuples:
dataset.get_columns(as_tuple=True)

# %%
# You can also restrict the column listing to a subset of variables:
dataset.get_columns(["c", "y"])

# %%
# ## Step 4 — Rename groups and variables
#
# Renaming a group is straightforward:
dataset.rename_group("G1", "foo")
dataset.group_names

# %%
# Renaming a variable in a *specific* group avoids ambiguity
# when the same name appears in several groups.
# Here `"x"` exists in both `"G2"` and `"G3"`;
# passing `"G2"` renames it only there:
dataset.rename_variable("x", "bar", "G2")
dataset.rename_variable("y", "baz")
dataset.variable_names

# %%
# ## Step 5 — Transform data in place
#
# [transform_data()][gemseo.datasets.dataset.Dataset.transform_data]
# applies a function to the underlying NumPy array of a selection of data.
# The selection follows the same `group_names / variable_names / components / indices`
# filtering logic used throughout the API.
# Here we double the values of `"bar"`:
dataset.transform_data(lambda x: 2 * x, variable_names="bar")
dataset.get_view(variable_names="bar")

# %%
# ## Step 6 — Query the dataset with get_view()
#
# [get_view()][gemseo.datasets.dataset.Dataset.get_view]
# returns a *view* (a sub-dataset) filtered by group, variable, component and/or row index.
# Get all data for variables `"b"` and `"x"`:
dataset.get_view(variable_names=["b", "x"])

# %%
# Get the whole `"inputs"` group:
dataset.get_view("inputs")

# %%
# Combine filters: variables `"b"` and `"x"`, component `0` only:
dataset.get_view(variable_names=["b", "x"], components=[0])

# %%
# ## Step 7 — Update entries selectively
#
# [update_data()][gemseo.datasets.dataset.Dataset.update_data]
# replaces a slice of the dataset identified by
# group, variable, component and/or row index.
# Here we overwrite row index `1` of the `"inputs"` group:
dataset.update_data([[10, 10, 10]], "inputs", indices=[1])
dataset

# %%
# We can focus on a specific variable as well.
# Here, let's change the first row of the variable "a".
dataset.update_data([[5, 5]], variable_names="a", indices=[0])
dataset

# %%
# !!! note
#     If a variable name exist in different groups,
#     you can specify both `variable_names` and `group_names`.
#
# ## Step 8 — Export to a dictionary of NumPy arrays
#
# [to_dict_of_arrays()][gemseo.datasets.dataset.Dataset.to_dict_of_arrays]
# converts the dataset into nested or flat dictionaries of NumPy arrays,
# which is convenient for interoperability with other libraries.
#
# The default nested form `{group_name: {variable_name: array}}`:
dataset.to_dict_of_arrays()

# %%
# Pass `by_group=False` to obtain a flat dictionary `{variable_name: array}`.
# When a variable name appears in more than one group,
# the key becomes `"group_name:variable_name"` to avoid collisions:
dataset.to_dict_of_arrays(False)

# %%
# ## Step 9 — Use IODataset for input/output data
#
# [IODataset][gemseo.datasets.io_dataset.IODataset] is a subclass of
# [Dataset][gemseo.datasets.dataset.Dataset] that pre-defines two group names:
# [INPUT_GROUP][gemseo.datasets.io_dataset.IODataset.INPUT_GROUP]
# and [OUTPUT_GROUP][gemseo.datasets.io_dataset.IODataset.OUTPUT_GROUP].
# It is the recommended structure for supervised machine learning
# and sensitivity analysis workflows.
#
# Use the dedicated helpers to add individual variables ...
io_dataset = IODataset()
io_dataset.add_input_variable("a", [[1.0, 2.0], [4.0, 5.0]])
io_dataset.add_input_variable("b", [[3.0], [6.0]])
io_dataset.add_output_variable("c", [[-1.0], [-2.0]])
io_dataset.add_variable("x", [[10.0], [20.0]])
io_dataset

# %%
# ... or whole groups at once with
# [add_input_group()][gemseo.datasets.io_dataset.IODataset.add_input_group]
# and [add_output_group()][gemseo.datasets.io_dataset.IODataset.add_output_group]:
io_dataset = IODataset()
io_dataset.add_input_group(
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], ["a", "b"], {"a": 2, "b": 1}
)
io_dataset.add_output_group([[-1.0], [-2.0]], ["c"])
io_dataset.add_variable("x", [[10.0], [20.0]])
io_dataset

# %%
# Domain-specific properties give direct access to the variable names:
io_dataset.input_names, io_dataset.output_names

# %%
# and the full list of variable names across all groups:
io_dataset.variable_names

# %%
# The number of samples (i.e. entries) and their indices:
io_dataset.n_samples

# %%
io_dataset.samples

# %%
# Filtered views for each group are also available as dataset properties:
io_dataset.input_dataset

# %%
io_dataset.output_dataset

# %%
# ## Step 10 — Use OptimizationDataset for optimization histories
#
# [OptimizationDataset][gemseo.datasets.optimization_dataset.OptimizationDataset]
# is a subclass of [Dataset][gemseo.datasets.dataset.Dataset]
# that pre-defines group names for the typical quantities found in an
# optimization history:
# [DESIGN_GROUP][gemseo.datasets.optimization_dataset.OptimizationDataset.DESIGN_GROUP],
# [OBJECTIVE_GROUP][gemseo.datasets.optimization_dataset.OptimizationDataset.OBJECTIVE_GROUP],
# [CONSTRAINT_GROUP][gemseo.datasets.optimization_dataset.OptimizationDataset.CONSTRAINT_GROUP],
# [OBSERVABLE_GROUP][gemseo.datasets.optimization_dataset.OptimizationDataset.OBSERVABLE_GROUP],
# as well as
# [EQUALITY_CONSTRAINT_GROUP][gemseo.datasets.optimization_dataset.OptimizationDataset.EQUALITY_CONSTRAINT_GROUP]
# and
# [INEQUALITY_CONSTRAINT_GROUP][gemseo.datasets.optimization_dataset.OptimizationDataset.INEQUALITY_CONSTRAINT_GROUP].
#
# Use the dedicated helpers to add individual variables ...
opt_dataset = OptimizationDataset()
opt_dataset.add_design_variable("x", [[1.0, 2.0], [4.0, 5.0]])
opt_dataset.add_design_variable("z", [[3.0], [6.0]])
opt_dataset.add_objective_variable("f", [[-1.0], [-2.0]])
opt_dataset.add_constraint_variable("c", [[-0.5], [0.1]])
opt_dataset.add_observable_variable("o", [[-3.0], [8.0]])
opt_dataset.add_variable("a", [[10.0], [20.0]])
opt_dataset

# %%
# ... or whole groups at once with
# [add_design_group()][gemseo.datasets.optimization_dataset.OptimizationDataset.add_design_group],
# [add_objective_group()][gemseo.datasets.optimization_dataset.OptimizationDataset.add_objective_group],
# [add_constraint_group()][gemseo.datasets.optimization_dataset.OptimizationDataset.add_constraint_group]
# and [add_observable_group()][gemseo.datasets.optimization_dataset.OptimizationDataset.add_observable_group]:
opt_dataset = OptimizationDataset()
opt_dataset.add_design_group(
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], ["x", "z"], {"x": 2, "z": 1}
)
opt_dataset.add_objective_group([[-1.0], [-2.0]], ["f"])
opt_dataset.add_constraint_group([[-0.5], [0.1]], ["c"])
opt_dataset.add_observable_group([[-3.0], [8.0]], ["o"])
opt_dataset.add_variable("a", [[10.0], [20.0]])
opt_dataset

# %%
# !!! note
#     [OptimizationDataset][gemseo.datasets.optimization_dataset.OptimizationDataset]
#     instances are created automatically by GEMSEO;
#     manual creation is shown here for illustration purposes only.
#
# Domain-specific properties give direct access to the variable names
# for each category:
opt_dataset.design_variable_names

# %%
opt_dataset.objective_names, opt_dataset.constraint_names, opt_dataset.observable_names

# %%
# and the full list of variable names across all groups:
opt_dataset.variable_names

# %%
# The number of iterations (i.e. entries) and their indices:
opt_dataset.n_iterations

# %%
opt_dataset.iterations

# %%
# Filtered views for each group are also available as dataset properties:
opt_dataset.design_dataset

# %%
opt_dataset.objective_dataset

# %%
opt_dataset.constraint_dataset

# %%
opt_dataset.observable_dataset

# %%
# ## Key takeaways
#
# - A [Dataset][gemseo.datasets.dataset.Dataset] is a pandas MultiIndex DataFrame
#   whose columns follow the three-level index `(GROUP, VARIABLE, COMPONENT)`;
#   build it with `add_variable()` or `add_group()` rather than the raw constructor.
# - The pair `(group_name, variable_name)` is the *variable identifier*;
#   the same variable name can live in several groups simultaneously.
# - `get_view()`, `update_data()` and `transform_data()` all accept the same
#   `group_names / variable_names / components / indices` filtering arguments,
#   giving you a consistent API for reading, writing and transforming data.
# - [IODataset][gemseo.datasets.io_dataset.IODataset] and
#   [OptimizationDataset][gemseo.datasets.optimization_dataset.OptimizationDataset]
#   are drop-in specializations of `Dataset` that pre-define domain-specific group names
#   and expose convenience helpers (`add_input_variable()`, `add_design_group()`, ...)
#   as well as filtered views (`input_dataset`, `objective_dataset`, ...).
# - All three classes are interoperable: `add_variable()`, `get_view()`,
#   `to_dict_of_arrays()` and the rest of the base API work unchanged on subclass instances.
#
# ## How-to guides
#
# For further information,
# please refer to the following how-to guides:
#
# - [Convert a database to a dataset][convert-a-database-to-a-dataset],
# - [Convert a cache to a dataset][convert-a-cache-to-a-dataset],
