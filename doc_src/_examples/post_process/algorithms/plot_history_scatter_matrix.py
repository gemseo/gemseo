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
Scatter plot matrix
===================

In this example, we illustrate the use of the :class:`.ScatterPlotMatrix`
post-processing on the Sobieski's SSBJ problem.

The :class:`.ScatterPlotMatrix` post-processing provide the scatter plot matrix among
design variables and outputs functions. Each non-diagonal block represents the samples
according to the x- and y- coordinates names while the diagonal ones approximate
the probability distributions of the variables, using a kernel-density estimator.
"""

from __future__ import annotations

from gemseo import execute_post
from gemseo.settings.post import ScatterPlotMatrix_Settings

execute_post(
    "sobieski_mdf_scenario.h5",
    settings_model=ScatterPlotMatrix_Settings(
        variable_names=["x_shared", "x_1", "x_2", "x_3", "-y_4"],
        save=False,
        show=True,
    ),
)
