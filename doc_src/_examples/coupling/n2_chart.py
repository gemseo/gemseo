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
Multidisciplinary coupling graph
================================
"""
from __future__ import annotations

from gemseo.api import generate_n2_plot
from gemseo.core.discipline import MDODiscipline
from numpy import ones

#######################################################################################
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
    inputs = {name: data for name in inputs}
    outputs = {name: data for name in outputs}
    discipline = MDODiscipline(discipline_name)
    discipline.input_grammar.update_from_data({name: data for name in inputs})
    discipline.output_grammar.update_from_data({name: data for name in outputs})
    disciplines.append(discipline)

#######################################################################################
# Generate the N2 chart
# ---------------------
# We do not want to save the N2 chart as a PNG or a PDF file,
# but open a browser, display it and handle it.
generate_n2_plot(disciplines, save=False, open_browser=True)

#######################################################################################
# `Click here <../../_static/n2.html>`_ to see the rendering.
