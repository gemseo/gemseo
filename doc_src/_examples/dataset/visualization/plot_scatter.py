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
Scatter
=======

"""
from __future__ import annotations

from gemseo.api import configure_logger
from gemseo.core.dataset import Dataset
from gemseo.post.dataset.scatter import Scatter
from numpy import linspace
from numpy import pi
from numpy import sin

configure_logger()


############################################################################
# Build a dataset
# ---------------
inputs = linspace(0, 1, 20)[:, None]
outputs = sin(2 * pi * inputs)
color = ["b" if abs(output) > 0.5 else "r" for output in outputs]

dataset = Dataset()
dataset.add_variable("x", inputs, "inputs")
dataset.add_variable("y", outputs, "outputs", cache_as_input=False)

############################################################################
# Plot y vs x
# -----------
# We can use the :class:`.Scatter` plot
plot = Scatter(dataset, "x", "y")
plot.color = color
plot.execute(save=False, show=True)
