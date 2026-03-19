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
"""# Use a Pydantic grammar

## Problem

You want to define fine-grained validation rules
for your discipline's Inputs/Outputs,
such as type constraints, value ranges, or custom validators.

## Solution

You can change the default discipline grammars and use Pydantic grammars instead.
Your discipline shall inherit from [Discipline][gemseo.core.discipline.Discipline].

!!! note
    Different grammars are supported, and this how-to can be adapted to use any of them.

## Step-by-step guide

"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import array
from pydantic import BaseModel
from pydantic import Field

from gemseo.core.discipline import Discipline
from gemseo.core.grammars.pydantic_grammar import PydanticGrammar
from gemseo.utils.pydantic_ndarray import NDArrayPydantic

if TYPE_CHECKING:
    from gemseo.typing import StrKeyMapping

# %%
# ### 1. Create the Pydantic model
#
# A Pydantic model is a class deriving from `pydantic.BaseModel`
# that describes the names and types of the data to be validated.
# Default values can be defined naturally, or with a `Field` which also allows to set a description useful for end users.
# Mind that default values with a mutable object may have to be defined with the
# `default_factory` argument of `Field`, see the Pydantic documentation for more.
# By default,
# Pydantic does not handle the typing of NumPy arrays.
# To support it,
# a special type shall be used, `NDArrayPydantic`.
# Like the standard NumPy type for `ndarray`, `NDArray`,
# this type can be specialized with the dtype like `NDArrayPydantic[int]`.


class Model(BaseModel):
    """The description of the model."""

    a_int: int
    """The description of an integer."""

    an_ndarray: NDArrayPydantic
    """The description of an ndarray."""

    an_ndarray_of_int: NDArrayPydantic[int]
    """The description of an ndarray of integers."""

    an_ndarray_with_default: NDArrayPydantic = Field(default_factory=lambda: array([0]))
    """The description of an ndarray with a default value."""

    a_str_with_default: str = "default"
    """The description of a string with a default value."""


# %%
# !!! Tip
#     Build complex grammars by subclassing a Pydantic model
#     rather than defining everything in a single class.
#
# ### 2. Create the grammar
#
input_grammar = PydanticGrammar("grammar", model=Model)
input_grammar


# %%
# !!! tip "Good practice"
#     `PydanticGrammar` should be defined in the `__init__` method of your discipline.
#
# ### 3. Create your discipline
#
class MyDiscipline(Discipline):
    """A discipline using a Pydantic grammar for inputs."""

    default_grammar_type = Discipline.GrammarType.PYDANTIC

    def __init__(self, name: str = "") -> None:
        super().__init__(name)
        self.input_grammar = input_grammar

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        return {}


discipline = MyDiscipline()
discipline
# %%
# And see the grammars
discipline.input_grammar

# %%
discipline.output_grammar

# %%
# ## Summary
#
# The grammar of your discipline can be changed by using the ``default_grammar_type``
# and by updating the input / output grammars.
