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
Parallel coordinates
====================

In this example, we illustrate the use of the :class:`.ParallelCoordinates` plot
on the Sobieski's SSBJ problem.

The :class:`.ParallelCoordinates` post-processing provides parallel coordinates plots
among design variables, outputs functions and constraints.

The :class:`.ParallelCoordinates` portrays the design variables history during the
scenario execution. Each vertical coordinate is dedicated to a design variable,
normalized by its bounds.

A polyline joins all components of a given design vector and is colored by objective
function values. This highlights the correlations between the values of the design
variables and the values of the objective function.
"""

from __future__ import annotations

from gemseo import execute_post
from gemseo.settings.post import ParallelCoordinates_Settings

execute_post(
    "sobieski_mdf_scenario.h5",
    settings_model=ParallelCoordinates_Settings(save=False, show=True),
)
