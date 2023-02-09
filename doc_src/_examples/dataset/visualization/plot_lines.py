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
Lines
=====

"""
from __future__ import annotations

from gemseo.api import configure_logger
from gemseo.core.dataset import Dataset
from gemseo.post.dataset.lines import Lines
from numpy import cos
from numpy import linspace
from numpy import pi
from numpy import sin

configure_logger()


############################################################################
# Build a dataset
# ---------------
inputs = linspace(0, 1, 10)[:, None]
outputs_1 = sin(2 * pi * inputs)
outputs_2 = cos(2 * pi * inputs)

dataset = Dataset()
dataset.add_variable("x", inputs, "inputs")
dataset.add_variable("y1", outputs_1, "outputs", cache_as_input=False)
dataset.add_variable("y2", outputs_2, "outputs", cache_as_input=False)

############################################################################
# Plot y1 and y2
# --------------
# We can use the :class:`.Lines` plot.
plot = Lines(dataset, variables=["y1", "y2"])
plot.linestyle = ["--", "-"]
plot.execute(save=False, show=True)
