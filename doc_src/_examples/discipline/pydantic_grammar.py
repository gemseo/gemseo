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
Use a pydantic grammar
======================
"""
from __future__ import annotations

from gemseo.core.grammars.errors import InvalidDataError
from gemseo.core.grammars.pydantic_grammar import PydanticGrammar
from numpy import array
from numpy import ndarray
from numpy._typing import NDArray
from pydantic import BaseModel
from pydantic import Field


# %%
# Create the pydantic model
# -------------------------
#
# The pydantic model is a class that describes the names and types of data to be
# validated.
# Descriptions are defined with docstrings, default values can be defined naturally.
# Mind that default values with a mutable object must be defined with the
# ``default_factory`` of a ``Field``.


class Model(BaseModel):
    """The description of the model."""

    a_int: int
    """The description of a_int."""

    an_ndarray: ndarray
    """The description of an_ndarray."""

    an_ndarray_of_int: NDArray[int]
    """The description of an_ndarray_of_int."""

    an_ndarray_with_default: ndarray = Field(default_factory=lambda: array([0]))
    """The description of an_ndarray_with_default."""

    a_str_with_default: str = "default"
    """The description of a_str."""


# %%
# Create the grammar
# ------------------
grammar = PydanticGrammar("grammar", model=Model)

# %%
# Show the contents of the grammar.
print(repr(grammar))
print()

# %%
# Validate data against the grammar
# ---------------------------------

# %%
# Validating missing data will raise an error shows the missing required elements,
# here the first 3 elements are missing.
try:
    grammar.validate({})
except InvalidDataError as error:
    print(error)

print()

# %%
# Validating data with bad type will raise an error shows the bad elements,
# here the first elements shall be an int and the third one shall be a ndarray of int.
try:
    grammar.validate(
        {"a_int": 0.0, "an_ndarray": array([1]), "an_ndarray_of_int": array([1.0])}
    )
except InvalidDataError as error:
    print(error)

print()

# %%
# Validating compliant data.
grammar.validate(
    {"a_int": 0, "an_ndarray": array([1]), "an_ndarray_of_int": array([1])}
)

# %%
# Grammar defaults
# ----------------
# As compared to the other types of grammars, the grammar defaults are be defined
# in the pydantic model and does not require to be manually defined from the grammar.
print(grammar.defaults)
print()

# %%
# Model inheritance
# -----------------
# Since pydantic models are classes, one can easily build grammar via inheritance of the
# pydantic model.
# Here we change the type of one element, and we add a new one.


class Model2(Model):
    """A model that inherits from a parent model."""

    an_ndarray: NDArray[float] = Field(default_factory=lambda: array([1.0]))
    """The new description of an_ndarray."""

    a_bool: bool = True
    """The description of a_bool."""


grammar = PydanticGrammar("grammar", model=Model2)

# %%
# Show the contents of the grammar.
print(repr(grammar))
print()
