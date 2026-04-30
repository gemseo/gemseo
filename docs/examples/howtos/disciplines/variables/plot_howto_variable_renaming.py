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

"""# Renaming variables

## Problem

In the context of a multidisciplinary study,
the disciplines must use a common naming convention
so that GEMSEO can automatically connect them.

Unfortunately,
model suppliers sometimes use different conventions.
For example,
the output of one model and the input of another model
represent the same parameter in the study,
but are called differently.

## Solution

The [gemseo.utils.discipline][gemseo.utils.discipline] module includes capabilities
enabling these models to be connected automatically using a set of translations.

!!! warning
    These features have an impact of discipline grammars.

## Step-by-step guide
"""

from __future__ import annotations

from numpy import array

from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.utils.discipline import VariableRenamer
from gemseo.utils.discipline import VariableTranslation
from gemseo.utils.discipline import rename_discipline_variables

# %%
# ### 1. Create the disciplines
#
# Let us consider four analytic disciplines.
# There is the first discipline, named `"A"`,
# for which we would like to rename `"a"` to `"x"` and `"c"` to `"z"`:
disciplines = [AnalyticDiscipline({"c": "2*a"}, name="A")]
# %%
# the second discipline, named `"C"`, to be used as is:
disciplines.append(AnalyticDiscipline({"t": "3*g"}, name="C"))
# %%
# the third disciplines, also named `"A"`,
# for which we would like to rename `"a"` to `"x"` and `"c"` to `"z"`
# (as said above):
disciplines.append(AnalyticDiscipline({"c": "4*a"}, name="A"))
# %%
# and the last one, named `"B"`,
# for which we would like to rename `"b"` to `"y"`:
disciplines.append(AnalyticDiscipline({"b": "5*j"}, name="B"))

# %%
# ### 2. Variable translation
#
# First,
# we need to introduce the notion of
# [VariableTranslation][gemseo.utils.discipline.VariableTranslation]
# to translate a discipline variable name according to a global taxonomy:
variable_translation = VariableTranslation(
    discipline_name="A", variable_name="a", new_variable_name="x"
)
variable_translation
# %%
# This object will be used to create a translator.
#
# ### 3. Create translators
#
# A [VariableRenamer][gemseo.utils.discipline.VariableRenamer] can be created
# from translations
# that can include both
# [VariableTranslation][gemseo.utils.discipline.VariableTranslation] instances
# and tuples of the form `(discipline_name, variable_name, new_variable_name)`:
renamer = VariableRenamer()
renamer.add_translation(variable_translation)
renamer.add_translation(("B", "b", "y"))
renamer.add_translation(
    VariableTranslation(discipline_name="A", variable_name="c", new_variable_name="z")
)
renamer


# %%
#
# !!! tips
#     There are several ways
#     to create a [VariableRenamer][gemseo.utils.discipline.VariableRenamer]:
#
#       - [add_translations_by_variable()][gemseo.utils.discipline.VariableRenamer.add_translations_by_variable]
#       - [add_translations_by_discipline()][gemseo.utils.discipline.VariableRenamer.add_translations_by_discipline]
#       - [from_dictionary()][gemseo.utils.discipline.VariableRenamer.from_dictionary]
#       - [from_translations()][gemseo.utils.discipline.VariableRenamer.from_translations]
#       - [from_csv()][gemseo.utils.discipline.VariableRenamer.from_csv]
#       - [from_spreadsheet()][gemseo.utils.discipline.VariableRenamer.from_spreadsheet]
#
# You can assess the translators with the property
renamer.translators

# %%
# !!! note
#     You may avoid creating a [VariableRenamer][gemseo.utils.discipline.VariableRenamer]
#     if you are able to create a nested dictionary from scratch:
#     `{"discipline_name": {old_variable_name: new_variable_name}}`.
#
#     However, creating such nested dictionary can be painful
#     when there are a lot of disciplines and variables to rename.

# %%
# ### 4. Rename discipline variables from translators
#
# [rename_discipline_variables()][gemseo.utils.discipline.rename_discipline_variables]
# is a function to rename some discipline variables from a translator
rename_discipline_variables(disciplines, renamer.translators)

# %%
# You may verify that the renaming has been done correctly:
disc_a, disc_c, other_disc_a, disc_b = disciplines
disc_a.execute({"x": array([3.0])})

# %%
disc_c.execute({"g": array([3.0])})

# %%
other_disc_a.execute({"x": array([3.0])})

# %%
disc_b.execute({"j": array([3.0])})

# %%
# ## Summary
#
# Variables can be renamed by using:
#
# - a [VariableRenamer][gemseo.utils.discipline.VariableRenamer] — a collection of translations, creatable from:
#
#     - Individual VariableTranslation objects or tuples ([from_translations()][gemseo.utils.discipline.VariableRenamer.from_translations])
#     - A nested dict like {"A": {"a": "x"}} ([from_dictionary()][gemseo.utils.discipline.VariableRenamer.from_dictionary])
#     - A CSV file ([from_csv()][gemseo.utils.discipline.VariableRenamer.from_csv])
#     - A spreadsheet ([from_spreadsheet()][gemseo.utils.discipline.VariableRenamer.from_spreadsheet])
#
# - [rename_discipline_variables()][gemseo.utils.discipline.rename_discipline_variables] which applies the renamer's translations to a list of disciplines in-place.
#   This changes the grammar of the discipline.
#
# Adding translations after creation supports three patterns:
# one-by-one ([add_translation()][gemseo.utils.discipline.VariableRenamer.add_translation]),
# by discipline ([add_translations_by_discipline()][gemseo.utils.discipline.VariableRenamer.add_translations_by_discipline]),
# or by variable across multiple disciplines ([add_translations_by_variable()][gemseo.utils.discipline.VariableRenamer.add_translations_by_variable]).
