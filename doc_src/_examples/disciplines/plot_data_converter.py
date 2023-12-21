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
Use a data converter
====================
"""

from __future__ import annotations

from gemseo.core.data_converters.json import JSONGrammarDataConverter
from gemseo.core.grammars.json_grammar import JSONGrammar

# %%
# Why?
# ----
#
# By default,
# the types of the coupling variables can be either numbers or 1D NumPy array.
# With custom data converters,
# one can add support to other types.
# For instance,
# to support 2D NumPy arrays,
# the following specialized data converter could be derived.


class DataConverter(JSONGrammarDataConverter):
    """A data converter where some coupling variables are 2D NumPy arrays."""

    # The names of the coupling variables that are 2D NumPy arrays.
    NAMES_WITH_2D_ARRAY = ("a_coupling_variable_name", "another_coupling_variable_name")

    # The shape of the 2D NumPy arrays.
    SHAPE = (2, 2)

    def convert_value_to_array(self, name, value):
        if name in self.NAMES_WITH_2D_ARRAY:
            return value.flatten()
        return super().convert_value_to_array(name, value)

    def convert_array_to_value(self, name, array_):
        if name in self.NAMES_WITH_2D_ARRAY:
            return array_.reshape(self.SHAPE)
        return super().convert_array_to_value(name, array_)


# %%
# Use the converter
# ----
#
# In order to be used,
# the custom data converter shall be set to the grammar class.
# The above custom data converter is defined for grammars of type JSONGrammar since
# it derives from JSONGrammarDataConverter.
# Declare the custom data converter with

JSONGrammar.DATA_CONVERTER_CLASS = DataConverter

# This shall be done the earliest,
# before any grammar is instantiated.
