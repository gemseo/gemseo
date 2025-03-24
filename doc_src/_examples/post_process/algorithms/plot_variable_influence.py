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
Variables influence
===================

In this example, we illustrate the use of the :class:`.VariableInfluence` plot
on the Sobieski's SSBJ problem.

The :class:`.VariableInfluence` post-processing performs first-order variable
influence analysis.

The method computes
:math:`\frac{d f}{d x_i} \cdot \left(x_{i_*} - x_{initial_design}\right)`, where
:math:`x_{initial_design}` is the initial value of the
variable and :math:`x_{i_*}` is the optimal value of the variable.
"""

from __future__ import annotations

from gemseo import execute_post
from gemseo.settings.post import VariableInfluence_Settings

execute_post(
    "sobieski_mdf_scenario.h5",
    settings_model=VariableInfluence_Settings(save=False, show=True),
)
