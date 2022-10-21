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
Rename the input and output variables
=====================================

The :class:`.RemappingDiscipline` can be used
to rename the input and output variables of an original discipline
including defining a variable as a part of an original one.
"""
from __future__ import annotations

from gemseo.core.discipline import MDODiscipline
from gemseo.disciplines.remapping import RemappingDiscipline
from numpy import array

# %%
# Let us consider a discipline that sums up the fruits of the market.
# The fruits can be classified into three categories:
# pears, Gala apples and Fuji apples.
# Then,
# the input variable of the discipline called ``fruits`` is a triplet
# containing the numbers of pears, Gala apples and Fuji apples so ordered.
# Concerning the outputs,
# ``n_fruits`` is the total number of fruits
# while ``n_fruits_per_category`` gathers the numbers of pears and apples so ordered.
# This discipline can be coded as follows:


class FruitCounting(MDODiscipline):
    def __init__(self) -> None:
        super().__init__()
        self.input_grammar.update(["fruits"])
        self.output_grammar.update(["n_fruits", "n_fruits_per_category"])
        self.default_inputs = {"fruits": array([1, 2, 3])}

    def _run(self) -> None:
        fruits = self.local_data["fruits"]
        self.store_local_data(
            n_fruits=array([fruits.sum()]),
            n_fruits_per_category=array([fruits[0], fruits[1:3].sum()]),
        )


# %%
# and we can instantiate it:
fruit_counting = FruitCounting()

# %%
# Then,
# we create a new discipline renaming ``fruits`` as ``pear`` and ``apples``
# and ``n_fruits`` and ``n_fruits_per_category`` as ``total`` and ``sub_total``
# to improve the naming:
clearer_fruit_counting = RemappingDiscipline(
    fruit_counting,
    {"pear": ("fruits", 0), "apples": ("fruits", [1, 2])},
    {"total": "n_fruits", "sub_total": "n_fruits_per_category"},
)

# %%
# Note:
#     :class:`.RemappingDiscipline` requires an instance of the original discipline,
#     the input names mapping to the original input names
#     and the outputs names mapping to the original output names.
#     More precisely,
#     an input or output name mapping looks like
#     ``{"new_x": "x", "new_y": ("y", components)}``
#     where the variable ``"new_x"`` corresponds to the original variable ``"x"``
#     and the variable ``"new_y"`` corresponds to some ``components``
#     of the original variable ``"y"``.
#     ``components`` can be an integer ``i`` (the ``i``-th component of ``y``),
#     a sequence of integers ``[i, j, k]``
#     (the ``i``-th, ``j``-th and ``k``-th components of ``y``)
#     or an iterable of integers ``range(i, j+1)``
#     (from the ``i``-th to the ``j``-th components of ``y``).

# %%
# We can execute this discipline with the original default input values,
# namely 1 pear, 2 Gala apples and 3 Fuji apples:
clearer_fruit_counting.execute()
print(clearer_fruit_counting.get_input_data())
print(clearer_fruit_counting.get_output_data())

# %%
# or with new input data:
clearer_fruit_counting.execute({"pear": array([4]), "apples": array([3, 1])})
print(clearer_fruit_counting.get_input_data())
print(clearer_fruit_counting.get_output_data())

# %%
# To be even more clear,
# we can split ``apples`` into ``gala`` and ``fuji``
# and ``sub_total`` into ``n_pears`` and ``n_apples``:
even_clearer_fruit_counting = RemappingDiscipline(
    clearer_fruit_counting,
    {"pear": "pear", "gala": ("apples", 0), "fuji": ("apples", 0)},
    {
        "total": "total",
        "n_pears": ("sub_total", 0),
        "n_apples": ("sub_total", 1),
    },
)

# %%
# and count the number of fruits:
even_clearer_fruit_counting.execute(
    {"pear": array([4]), "gala": array([3]), "fuji": array([1])}
)
print(even_clearer_fruit_counting.get_input_data())
print(even_clearer_fruit_counting.get_output_data())
