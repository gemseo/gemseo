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
"""# Tutorial - Choose among alternatives with catalog variables

## Goal

In this tutorial,
you will describe a design choice that is neither a number nor an ordered value:
the material of a panel, picked among aluminium, steel and titanium.
You will gather these alternatives and their properties in a
[Catalog][gemseo.space.catalog.catalog.Catalog],
declare a catalog variable in a [DesignSpace][gemseo.space.design.DesignSpace]
whose value is the position of the chosen alternative,
and save the design space to disk.
Keep in mind what a catalog variable cannot do yet:
no optimizer and no DOE algorithm handles it.
"""

from __future__ import annotations

from pandas import DataFrame

from gemseo.enum import DesignVariableType
from gemseo.space import Catalog
from gemseo.space import DesignSpace

# %%
# ## Step 1 — Why a catalog variable?
#
# The designer of a panel hesitates between three materials:
#
# | Material  | Density (t/m³) | Cost (€/kg) |
# |-----------|---------------:|------------:|
# | aluminium | 2.7            | 10.0        |
# | steel     | 7.8            | 5.0         |
# | titanium  | 4.5            | 50.0        |
#
# None of the other variable types fits this choice.
# A real or an integer variable implies an order and a distance:
# steel is not "between" aluminium and titanium.
# A discrete variable takes a few numbers,
# but a material is not a number,
# it is a row of properties.
#
# A catalog variable handles this case:
# it chooses one row of a table,
# and its value is the **position** of that row, from 0 to 2 here.
# The model then reads the properties of the chosen row.

# %%
# ## Step 2 — Build a catalog
#
# A catalog has one row per alternative,
# one column per property
# and one label naming each alternative.
# It can be built from a mapping from a property name to a sequence of values:
materials = Catalog(
    properties={"density": [2.7, 7.8, 4.5], "cost": [10.0, 5.0, 50.0]},
    labels=["aluminium", "steel", "titanium"],
)
materials.to_dataframe()

# %%
# The catalog **formats** its input:
# it strips the strings of their leading and trailing whitespace,
# and drops, with a warning, a property holding no value.
# Here, the label `" steel "` becomes `"steel"`
# and the empty property `"comment"` is dropped:
Catalog(
    properties={
        "density": [2.7, 7.8, 4.5],
        "cost": [10.0, 5.0, 50.0],
        "comment": ["", "", ""],
    },
    labels=["aluminium", " steel ", "titanium"],
).to_dataframe()

# %%
# A catalog can also be built from a `DataFrame`,
# whose index gives the labels.
# This is handy when the alternatives come from a spreadsheet or a database,
# e.g. with `pandas.read_csv`:
suppliers = Catalog(
    properties=DataFrame({"delay": [3.0, 10.0]}, index=["local", "overseas"])
)
suppliers.to_dataframe()

# %%
# The contents are read back through the labels and the properties:
materials.labels

# %%
materials.properties["density"]

# %%
# A catalog is **immutable**:
# neither the input passed to it
# nor the arrays and tables it hands out can change it.
# Changing the alternatives means building a new catalog.

# %%
# ## Step 3 — Add catalog variables to a design space
#
# [add_catalog_variable()][gemseo.space.design.DesignSpace.add_catalog_variable]
# declares a catalog variable from a catalog
# and, optionally, its current alternative,
# given by its position or by its label.
# The panel is steel and 2 mm thick:
design_space = DesignSpace()
design_space.add_real_variable("thickness", lower_bound=1.0, upper_bound=5.0, value=2.0)
design_space.add_catalog_variable("material", materials, value="steel")

# %%
# The label is replaced by the position of the alternative it names, here `1`;
# `value=1` would have done the same.
# A label naming no alternative is rejected,
# and so is a label naming several ones,
# since a catalog accepts duplicate labels:
# pass the position of the alternative then.
design_space.get_current_value(["material"])

# %%
# The catalog can also be passed as its properties,
# the design space building the catalog itself.
# Without a value,
# [initialize_missing_current_values()][gemseo.space.design.DesignSpace.initialize_missing_current_values]
# selects the first alternative:
design_space.add_catalog_variable(
    "supplier", DataFrame({"delay": [3.0, 10.0]}, index=["local", "overseas"])
)
design_space.initialize_missing_current_values()
design_space

# %%
# The bounds of a catalog variable are 0 and the number of alternatives minus one;
# they are derived from the catalog and cannot be set.
# A catalog variable is scalar:
# a panel made of two materials needs two catalog variables.
#
# The catalog is read back from the variable:
design_space.variables["material"].catalog.to_dataframe()

# %%
# and the type of the variables tells whether the design space holds a catalog variable:
design_space.variables.has_variables_of_type(DesignVariableType.CATALOG)

# %%
# ## Step 4 — Save and reload the design space
#
# A catalog does not fit in a CSV cell,
# so [to_csv()][gemseo.space.design.DesignSpace.to_csv] refuses a design space
# holding a catalog variable:
try:
    design_space.to_csv("design_space.csv")
except ValueError as error:
    message = str(error)

message

# %%
# Use HDF instead:
# [to_hdf()][gemseo.space.design.DesignSpace.to_hdf] writes the catalogs
# and [from_hdf()][gemseo.space.design.DesignSpace.from_hdf] reads them back.
design_space.to_hdf("design_space.h5")
reloaded = DesignSpace.from_hdf("design_space.h5")
reloaded.variables["material"].catalog.to_dataframe()

# %%
# !!! warning
#     Since the stored values are positions in the catalogs,
#     appending to an HDF file, or loading a database from one,
#     requires the catalogs to be those stored in the file;
#     otherwise, a `ValueError` is raised and the file is left untouched.

# %%
# ## Key takeaways
#
# - A catalog variable chooses one alternative in a catalog,
#   a table with one row per alternative and one column per property.
# - Its value is the position of the alternative;
#   its bounds are derived from the catalog, which cannot be changed.
# - No optimizer and no DOE algorithm handles a catalog variable yet.
# - A design space holding a catalog variable is saved to HDF, not to CSV.
#
# ## How-to guides
#
# For further information,
# please refer to the following how-to guides:
#
# - [How to import and export a design space from disk][],
# - [How to reduce a design space][].
