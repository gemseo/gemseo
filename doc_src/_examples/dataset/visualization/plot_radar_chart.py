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

"""
from __future__ import annotations

from gemseo.api import configure_logger
from gemseo.core.dataset import Dataset
from gemseo.post.dataset.radar_chart import RadarChart
from numpy import array

configure_logger()


############################################################################
# Build a dataset
# ---------------
dataset = Dataset()
dataset.add_variable("x1", array([[0.2, 0.4, 0.5], [0.1, 0.3, 0.5]]))
dataset.add_variable("x2", array([[0.6], [0.5]]))
dataset.add_variable("x3", array([[0.8], [0.7]]))
dataset.row_names = ["series_1", "series_2"]

############################################################################
# Plot the two series on a radar chart
# ------------------------------------
# We can use the :class:`~gemseo.post.dataset.radar_chart.RadarChart` plot:
plot = RadarChart(dataset, connect=True, radial_ticks=True)
plot.rmin = -0.5
plot.rmax = 1.0
plot.execute(save=False, show=True)
