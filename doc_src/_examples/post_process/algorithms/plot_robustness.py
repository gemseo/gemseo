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
Robustness
==========

In this example, we illustrate the use of the :class:`.Robustness` plot
on the Sobieski's SSBJ problem.

The :class:`.Robustness` post-processing plots the robustness of the optimum in a box
plot. Using the quadratic approximations of all the output functions, we propagate
analytically a normal distribution with 1% standard deviation on all the design
variables, assuming no cross-correlations of inputs, to obtain the mean and standard
deviation of the resulting normal distribution. A series of samples are randomly
generated from the resulting distribution, whose quartiles are plotted, relatively to
the values of the function at the optimum. For each function (in abscissa), the plot
shows the extreme values encountered in the samples (top and bottom bars). Then, 95% of
the values are within the blue boxes. The average is given by the red bar.
"""

from __future__ import annotations

from gemseo import execute_post
from gemseo.settings.post import Robustness_Settings

execute_post(
    "sobieski_mdf_scenario.h5",
    settings_model=Robustness_Settings(save=False, show=True),
)
