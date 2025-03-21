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

from numpy import array

from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.utils.discipline import VariableRenamer
from gemseo.utils.discipline import VariableTranslation
from gemseo.utils.discipline import rename_discipline_variables

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
# The :mod:`gemseo.utils.discipline` module includes capabilities
# enabling these models to be connected automatically using a set of translations.
#
# Renaming discipline variables from translators
# ----------------------------------------------
# :func:`.rename_discipline_variables` is a function
# to rename some discipline variables from a dictionary of translators
# of the form ``{discipline_name: {variable_name: new_variable_name}}``.
# For example,
# let us consider four analytic disciplines.
# There is the first discipline, named ``"A"``,
# for which we would like to rename ``"a"`` to ``"x"`` and ``"c"`` to ``"z"``:
disciplines = [AnalyticDiscipline({"c": "2*a"}, name="A")]
# %%
# the second discipline, named ``"C"``, to be used as is:
disciplines.append(AnalyticDiscipline({"t": "3*g"}, name="C"))
# %%
# the third disciplines, also named ``"A"``,
# for which we would like to rename ``"a"`` to ``"x"`` and ``"c"`` to ``"z"``
# (as said above):
disciplines.append(AnalyticDiscipline({"c": "4*a"}, name="A"))
# %%
# and the last one, named ``"B"``,
# for which we would like to rename ``"b"`` to ``"y"``:
disciplines.append(AnalyticDiscipline({"b": "5*j"}, name="B"))
# %%
# The following nested dictionary indexed by the discipline names
# defines the translators:
translators = {"A": {"a": "x", "c": "z"}, "B": {"b": "y"}}
# %%
# Finally,
# we can rename the input and output variables of the disciplines:
rename_discipline_variables(disciplines, translators)
# %%
# and verify that the renaming has been done correctly:
disc_a, disc_c, other_disc_a, disc_b = disciplines
assert disc_a.execute({"x": array([3.0])})["z"] == array([6.0])
assert disc_c.execute({"g": array([3.0])})["t"] == array([9.0])
assert other_disc_a.execute({"x": array([3.0])})["z"] == array([12.0])
assert disc_b.execute({"j": array([3.0])})["y"] == array([15.0])
# %%
# .. tip::
#
#    Creating the nested dictionary ``translators`` can be a pain
#    when there is a lot of disciplines and a lot of variables to rename.
#    The following sections present some tools to facilitate its creation.
#
# Create translators easily
# -------------------------
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
# we can create a :class:`.VariableRenamer` from a set of translations
# that can include both :class:`.VariableTranslation` instances
# and tuples of the form ``(discipline_name, variable_name, new_variable_name)``:
renamer = VariableRenamer.from_translations(
    variable_translation,
    ("B", "b", "y"),
    VariableTranslation(discipline_name="A", variable_name="c", new_variable_name="z"),
)
renamer

# %%
# This object offers a :attr:`.translators` property
# mapping from discipline names to dictionaries,
# themselves mapping from the discipline variable names to the global variable names:
renamer.translators

# %%
# You can use these translators to rename the corresponding discipline variables,
# by calling the :func:`.rename_discipline_variables` function
# presented in the first section:
# ``rename_discipline_variables(disciplines, renamer.translators)``.

# %%
# You can also access the :attr:`.translations`:
renamer.translations

# %%
# Add translations
# ~~~~~~~~~~~~~~~~
# Finally,
# you can add translations to a :class:`.VariableRenamer`.
#
# One by one
# ..........
# You can add them one by one
# using its :meth:`~.VariableRenamer.add_translation` method:
renamer.add_translation(
    VariableTranslation(discipline_name="Y", variable_name="x", new_variable_name="b")
)
renamer.translations

# %%
# By discipline
# .............
# You can add several translations associated with the same discipline
# using its :meth:`~.VariableRenamer.add_translations_by_discipline` method
# from a discipline name
# and a dictionary of the form ``{variable_name: new_variable_name}``:
renamer.add_translations_by_discipline("T", {"x": "e", "o": "p"})
renamer.translations

# %%
# By variable
# ...........
# You add several translations,
# indicating which variables in which disciplines are to be renamed in the same way,
# using its :meth:`~.VariableRenamer.add_translations_by_variable` method
# from a new variable name
# and a dictionary of the form ``{discipline_name: variable_name}``:
renamer.add_translations_by_discipline("i", {"A": "r", "B": "m"})
renamer.translations

# %%
# Define the renamer from a dictionary
# ------------------------------------
# You can also use a nested dictionary
# of the form ``{discipline_name: {variable_name: new_variable_name}}``
# to create the :class:`.VariableRenamer`:
renamer = VariableRenamer.from_dictionary({"A": {"a": "x", "c": "z"}, "B": {"b": "y"}})
renamer.translators

# %%
# Define the renamer from a file
# ------------------------------
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
