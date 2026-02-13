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
"""# Generate a coupling graph.

In this example,
you will learn how to generate a coupling graph from a list of disciplines.

## Problem

You have multiple disciplines.
You want to visualize their couplings as a graph.

## Solution

You need to create a coupling graph,
using the [generate_coupling_graph()][gemseo.generate_coupling_graph] function.

## Step-by-step guide
"""

from __future__ import annotations

from gemseo import generate_coupling_graph
from gemseo.utils.discipline import DummyDiscipline

# %%
# ### 1. Create the disciplines
#
# In this example, you will use dummy disciplines that do nothing, but the same applies to any GEMSEO disciplines:
dummy_disciplines = [
    DummyDiscipline(name=name, input_names=input_names, output_names=output_names)
    for (name, input_names, output_names) in (
        ("A", ["a"], ["b"]),
        ("B", ["c"], ["a", "n"]),
        ("C", ["b", "d"], ["c", "e"]),
        ("D", ["f"], ["d", "g"]),
        ("E", ["e"], ["f", "h", "o"]),
        ("F", ["g", "j"], ["i"]),
        ("G", ["i", "h"], ["k", "l"]),
        ("H", ["k", "m"], ["j"]),
        ("I", ["l"], ["m", "w"]),
        ("J", ["n", "o"], ["p", "q"]),
        ("K", ["y"], ["x"]),
        ("L", ["w", "x"], ["y", "z"]),
        ("M", ["p", "s"], ["r"]),
        ("N", ["r"], ["t", "u"]),
        ("O", ["q", "t"], ["s", "v"]),
        ("P", ["u", "v", "z"], ["z"]),
    )
]

# %%
# ### 2. Generate the coupling graph
#
# The coupling graph represents each discipline by a node
# and each coupling variable by an edge.
# By default,
# the [generate_coupling_graph()][gemseo.generate_coupling_graph] function
# saves the graphical representation of the coupling graph as a PDF file.
# Here,
# You prefer to display it in this web page
# by setting `file_path` to `""`
# (the same would work in a Jupyter Notebook).
generate_coupling_graph(dummy_disciplines, file_path="")

# %%
# ### 3. Generate the condensed coupling graph
#
# You can also draw the condensed coupling graph,
# where each groups of strongly coupled disciplines is represented by a node:
generate_coupling_graph(dummy_disciplines, file_path="", full=False)

# %%
# ## Summary
#
# You can plot the coupling graph
# with the [generate_coupling_graph()][gemseo.generate_coupling_graph] function.
# You can also merge the highly coupled disciplines into a single node
# to make it clearer;
# the condensed coupling graph is obtained by setting the `full` argument to `False`.
