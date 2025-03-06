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
Pareto front
=============

In this example, we illustrate the use of the :class:`.ParetoFront` plot
on the Sobieski's SSBJ problem.

The :class:`.ParetoFront` post-processing provides a matrix of plots (if there are more
than 2 objectives). It indicates in red the locally non-dominated points for the current
objective, and in green the globally (all objectives) Pareto optimal points.
"""

from __future__ import annotations

from gemseo import execute_post
from gemseo.settings.post import ParetoFront_Settings

execute_post(
    "sobieski_mdf_scenario.h5",
    settings_model=ParetoFront_Settings(
        objectives=["g_3", "-y_4"],
        save=False,
        show=True,
    ),
)
