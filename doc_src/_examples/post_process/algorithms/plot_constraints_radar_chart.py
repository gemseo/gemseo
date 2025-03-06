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
Radar chart
===========

In this example, we illustrate the use of the :class:`.RadarChart` post-processing
on the Sobieski's SSBJ problem.

The :class:`.RadarChart` post-processing plots the constraints functions specified by
the user, at a given iteration, on a radar style chart.

By default, the last iteration is used. This post-processing of the constraints scales
better with the number of constraints than the constraint plot provided by the
:class:`.OptHistoryView`.
"""

from __future__ import annotations

from gemseo import execute_post
from gemseo.settings.post import RadarChart_Settings

execute_post(
    "sobieski_mdf_scenario.h5",
    settings_model=RadarChart_Settings(
        constraint_names=["g_1", "g_2", "g_3"],
        save=False,
        show=True,
    ),
)
