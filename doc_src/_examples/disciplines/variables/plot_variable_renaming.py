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
"""
Renaming variables
==================
"""

from __future__ import annotations

from gemseo.utils.variable_renaming import VariableRenamer
from gemseo.utils.variable_renaming import VariableTranslation

# %%
# In the context of a multidisciplinary study,
# the disciplines must use a common naming convention
# so that GEMSEO can automatically connect them.
#
# Unfortunately,
# model suppliers sometimes use different conventions.
# For example,
# the output of one model and the input of another model
# represent the same parameter in the study.
#
# The :mod:`.variable_renaming` module
# enables these models to be connected automatically using a set of translations.
#
# The main objects
# ----------------
# Variable translation
# ~~~~~~~~~~~~~~~~~~~~
# First,
# we need to introduce the notion of :class:`.VariableTranslation`
# to translate a discipline variable name according to a global taxonomy:
variable_translation = VariableTranslation(
    discipline_name="A", variable_name="a", new_variable_name="x"
)
variable_translation

# %%
# Variable renamer
# ~~~~~~~~~~~~~~~~
# Then,
# we can create a :class:`.VariableRenamer` from a set of translations:
renamer = VariableRenamer.from_translations(
    variable_translation,
    VariableTranslation(discipline_name="B", variable_name="b", new_variable_name="y"),
    VariableTranslation(discipline_name="A", variable_name="c", new_variable_name="z"),
)
renamer

# %%
# This object offers a :attr:`.translators` property
# mapping from discipline names to dictionaries,
# themselves mapping from the discipline variable names to the global variable names:
renamer.translators

# %%
# We can also access the translations:
renamer.translations

# %%
# Define the translation from tuples or dictionaries
# --------------------------------------------------
# The :class:`.VariableRenamer` can also be created from tuples
# of the form ``(discipline_name, variable_name, new_variable_name)``:
renamer = VariableRenamer.from_tuples(("A", "a", "x"), ("B", "b", "y"), ("A", "c", "z"))
renamer.translators
# %%
# or from a nested dictionary
# of the form ``{discipline_name: {variable_name: new_variable_name}}``:
renamer = VariableRenamer.from_dictionary({"A": {"a": "x", "c": "z"}, "B": {"b": "y"}})
renamer.translators

# %%
# Define the translation from a file
# ----------------------------------
# Lastly,
# The :class:`.VariableRenamer` can easily be created from a file,
# which may be more convenient from a user point of view.
#
# From a CSV file
# ~~~~~~~~~~~~~~~
# Given a CSV file
# whose rows are translations
# and columns denote
# the name of the source discipline,
# the name of the variable within the source discipline,
# the name of the target discipline
# and the name of the variable with the target discipline:
#
# .. code-block:: console
#
#    A,a,x
#    B,b,y
#    A,c,z
#
# we can create this object with the following lines:
#
# .. code-block:: python
#
#    renamer = VariableRenamer.from_csv(file_path)
#    translators = renamer.translators
#
# .. tip::
#     Use the ``sep`` argument to use a separator character other than ``","``.
#
# From a spreadsheet file
# ~~~~~~~~~~~~~~~~~~~~~~~
# Given a spreadsheet file
# whose rows are translations
# and columns denote
# the name of the discipline,
# the name of the variable,
# and the new name of the discipline:
#
# .. figure:: /_images/renaming_spreadsheet.png
#
# we can create this object with the following lines:
#
# .. code-block:: python
#
#    renamer = VariableRenamer.from_spread_sheet(file_path)
#    translators = renamer.translators
