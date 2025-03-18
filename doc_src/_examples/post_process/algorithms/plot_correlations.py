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
r"""
Correlations
============

In this example, we illustrate the use of the :class:`.Correlations` post-processing
on the Sobieski's SSBJ problem.

A correlation coefficient indicates whether there is a linear relationship between two
quantities :math:`x` and :math:`y`. It is the normalized covariance between the two
quantities defined as:

.. math::

   R_{xy}=\frac {\sum \limits _{i=1}^n(x_i-{\bar{x}})(y_i-{\bar{y}})}{ns_{x}s_{y}}
   =\frac {\sum \limits _{i=1}^n(x_i-{\bar{x}})(y_i-{\bar{y}})}{\sqrt {\sum
   \limits _{i=1}^n(x_i-{\bar{x}})^{2}\sum \limits _{i=1}^n(y_i-{\bar{y}})^{2}}}

The :class:`.Correlations` post-processing provides scatter plots of correlated
variables among design variables, outputs functions, and constraints.

By default, only the variables with a correlation coefficient greater than 0.95 are
considered. The threshold value can be modified in the post-processing settings.
"""

from __future__ import annotations

from gemseo import execute_post
from gemseo.settings.post import Correlations_Settings

# Correlations of the constraint `g_3`.
execute_post(
    "sobieski_mdf_scenario.h5",
    settings_model=Correlations_Settings(
        func_names=["g_3"],
        coeff_limit=0.95,  # Default value, here for illustration purpose.
        save=False,
        show=True,
    ),
)
