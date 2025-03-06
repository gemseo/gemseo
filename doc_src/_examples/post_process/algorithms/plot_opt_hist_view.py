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
Optimization History View
=========================

In this example, we illustrate the use of the :class:`.OptHistoryView` post-processing
on the Sobieski's SSBJ problem.

The :class:`.OptHistoryView` post-processing creates a series of plots:

- **The design variables history**: shows the normalized values of the design variables.
  On the :math:`y` axis are the components of the design variables vector and on the
  :math:`x` axis the iterations. The values are **normalized** between 0 and 1 and
  represented by colors.
- **The objective function history**: shows the evolution of the objective function
  value during the optimization.
- **The distance to the best design variables**: shows the Euclidean distance
  :math:`||x-x^*||_2` between the different design variable vectors considered by the
  optimizer and the best one (in log scale).
- **The inequality constraint history**: shows the evolution of the constraints values.
  On the :math:`y` axis are the components of the constraints and on the :math:`x` axis
  the iterations.The color indicates whether the constraint component is satisfied: red
  means violated, white means active and green means satisfied.
  For an :ref:`IDF formulation <idf_formulation>`, an additional plot is created to
  track the equality constraint history.
"""

from __future__ import annotations

from gemseo import execute_post
from gemseo.settings.post import OptHistoryView_Settings

execute_post(
    "sobieski_mdf_scenario.h5",
    settings_model=OptHistoryView_Settings(
        variable_names=["x_1", "x_2"],
        save=False,
        show=True,
    ),
)
