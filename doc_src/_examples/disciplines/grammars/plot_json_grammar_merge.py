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
# Antoine DECHAUME
"""
Merge or update a JSONGrammar
=============================
"""

from __future__ import annotations

import contextlib
from copy import deepcopy

from gemseo.core.grammars.errors import InvalidDataError
from gemseo.core.grammars.json_grammar import JSONGrammar

# %%
# Create the JSON grammars
# ------------------------
#
# The first grammar has the elements `name1` and `name2` that shall be integers.
# The second grammar has the elements `name2` and `name3` that shall be strings.

grammar_1 = JSONGrammar("grammar_1")
grammar_1.update_from_file("grammar_1.json")

grammar_2 = JSONGrammar("grammar_2")
grammar_2.update_from_file("grammar_2.json")

# %%
# Keep a pristine copy of the first grammar such that we can experiment more than once
# on it.
grammar_1_copy = deepcopy(grammar_1)

# %%
# Update without merge
# --------------------
#
# By default, the update method acts like for a dictionary, i.e. an already existing
# key has its value overriden. Here the value for the element `name2` is now string.
grammar_1.update(grammar_2)
grammar_1

# %%
# On validation, the allowed type is only the one from the second grammar.
with contextlib.suppress(InvalidDataError):
    grammar_1.validate({"name2": 0})

grammar_1.validate({"name2": "a string"})

# %%
# Update with merge
# -----------------
#
# First use the initial grammar from its copy.
grammar_1 = grammar_1_copy

# %%
# When merging elements, an already existing element of the grammar will have both the
# old and the new value.
grammar_1.update(grammar_2, merge=True)
grammar_1

# %%
# On validation, the allowed types can be any of the types from the original grammars,
# here either integer or string.
grammar_1.validate({"name2": 0})
grammar_1.validate({"name2": "a string"})
