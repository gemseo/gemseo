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
Constraints history
===================

In this example, we illustrate the use of the :class:`.ConstraintsHistory`
post-processing on the Sobieski's SSBJ problem.

The :class:`.ConstraintsHistory` post-processing plots the history of the constraints
functions specified by the user with respect to the iteration. Each constraint history
is represented in a subplot where the background color gives qualitative indications
on the constraint violation: areas where the constraint is active are white, the ones
where it is violated ones are red and the ones where it is satisfied are green.

This post-processing provides more precise information on the constraints than
:class:`.OptHistoryView` but scales less with the number of constraints.
"""

from __future__ import annotations

from gemseo import execute_post
from gemseo.settings.post import ConstraintsHistory_Settings

execute_post(
    "sobieski_mdf_scenario.h5",
    settings_model=ConstraintsHistory_Settings(
        constraint_names=["g_1", "g_2", "g_3"],
        save=False,
        show=True,
    ),
)
