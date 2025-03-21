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
"""N2 chart.
========
"""

from __future__ import annotations

from gemseo import generate_n2_plot
from gemseo.utils.discipline import DummyDiscipline

# %%
# Create the disciplines
# ----------------------
# First,
# we create dummy disciplines that do nothing:
dummy_disciplines = [
    DummyDiscipline(name=name, input_names=input_names, output_names=output_names)
    for (name, input_names, output_names) in (
        ("A", [f"a{i}" for i in range(500)], ["b"]),
        ("B", ["c"], [f"a{i}" for i in range(500)] + ["n"]),
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
# Generate the N2 chart
# ---------------------
# The N2 chart is a tabular way to visualize multidisciplinary coupling variables.
# The disciplines are located on the diagonal of the chart
# while the coupling variables are situated on the other blocks of the matrix view.
# A coupling variable is outputted by a discipline horizontally
# and enters another vertically.
#
# In the classical representation,
# a blue diagonal block represents a self-coupled discipline,
# *i.e.* a discipline having some of its outputs as inputs.
#
# Because of its tabular structure,
# the N2 chart is hard to analyze when the number of disciplines increases.
# This is the reason why in this example,
# we propose to use an interactive representation in a web browser:
generate_n2_plot(dummy_disciplines, save=False, show_html=True)

# %%
# `Click here <../../_static/n2.html>`_ to see the rendering.
