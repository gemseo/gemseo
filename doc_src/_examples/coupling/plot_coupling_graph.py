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
Coupling graph
==============
"""

from __future__ import annotations

from numpy import ones

from gemseo import generate_coupling_graph
from gemseo.core.discipline import Discipline

# %%
# Create the disciplines
# ----------------------
descriptions = {
    "A": ([f"a{i}" for i in range(500)], ["b"]),
    "B": (["c"], [f"a{i}" for i in range(500)] + ["n"]),
    "C": (["b", "d"], ["c", "e"]),
    "D": (["f"], ["d", "g"]),
    "E": (["e"], ["f", "h", "o"]),
    "F": (["g", "j"], ["i"]),
    "G": (["i", "h"], ["k", "l"]),
    "H": (["k", "m"], ["j"]),
    "I": (["l"], ["m", "w"]),
    "J": (["n", "o"], ["p", "q"]),
    "K": (["y"], ["x"]),
    "L": (["w", "x"], ["y", "z"]),
    "M": (["p", "s"], ["r"]),
    "N": (["r"], ["t", "u"]),
    "O": (["q", "t"], ["s", "v"]),
    "P": (["u", "v", "z"], ["z"]),
}
disciplines = []
data = ones(1)
for discipline_name, (inputs, outputs) in descriptions.items():
    inputs = dict.fromkeys(inputs, data)
    outputs = dict.fromkeys(outputs, data)
    discipline = Discipline(discipline_name)
    discipline.input_grammar.update_from_data(dict.fromkeys(inputs, data))
    discipline.output_grammar.update_from_data(dict.fromkeys(outputs, data))
    disciplines.append(discipline)

# %%
# Generate the coupling graph
# ---------------------------
# The coupling graph represents each discipline by a node
# and each coupling variable by an edge.
# By default,
# the :func:`.generate_coupling_graph` function saves the graphical representation
# of the coupling graph as a PDF file.
# Here,
# we prefer to display it in this web page
# by setting ``file_path`` to ``""``
# (the same would work in a notebook).
generate_coupling_graph(disciplines, file_path="")

# %%
# We can also draw the condensed coupling graph,
# where each groups of strongly coupled disciplines is represented by a node:
generate_coupling_graph(disciplines, file_path="", full=False)
