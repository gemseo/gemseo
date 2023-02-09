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
Boxplot
=======
"""
from __future__ import annotations

from gemseo.api import configure_logger
from gemseo.core.dataset import Dataset
from gemseo.post.dataset.boxplot import Boxplot
from numpy import hstack
from numpy import linspace

configure_logger()


############################################################################
# Build a dataset
# ---------------
inputs = linspace(-1, 1, 100)[:, None]

dataset = Dataset(name="Foo")
dataset.add_variable("y1", inputs**2, "outputs", cache_as_input=False)
dataset.add_variable(
    "y2", hstack((inputs**3, inputs**4)), "outputs", cache_as_input=False
)

other_dataset = Dataset(name="Bar")
other_dataset.add_variable("y1", -(inputs**2), "outputs", cache_as_input=False)
other_dataset.add_variable(
    "y2", hstack((-(inputs**3), -(inputs**4))), "outputs", cache_as_input=False
)

############################################################################
# Plot y1 and y2
# --------------
# We can use the :class:`.Boxplot` plot.
plot = Boxplot(dataset)
plot.xlabel = "Variables"
plot.ylabel = "Values"
plot.title = "Standard boxplots"
plot.execute(save=False, show=True)

############################################################################
# Plot with centering
# -------------------
# We can center the data:
plot = Boxplot(dataset, center=True)
plot.title = "With centering"
plot.execute(save=False, show=True)

############################################################################
# Plot with scaling
# -----------------
# We can scale the data (normalization with the standard deviation):
plot = Boxplot(dataset, scale=True)
plot.title = "With scaling"
plot.execute(save=False, show=True)

############################################################################
# Plot without outliers
# ---------------------
# We can remove the outliers:
plot = Boxplot(dataset, add_outliers=False)
plot.title = "Without outliers"
plot.execute(save=False, show=True)

############################################################################
# Plot with confidence intervals
# ------------------------------
# We can add confidence intervals for the median:
plot = Boxplot(dataset, add_confidence_interval=True)
plot.title = "Confidence intervals"
plot.execute(save=False, show=True)

############################################################################
# Plot horizontally
# -----------------
# We can use horizontal bars:
plot = Boxplot(dataset, use_vertical_bars=False)
plot.title = "Horizontal bars"
plot.execute(save=False, show=True)

############################################################################
# Plot with other datasets
# ------------------------
# We can add a dataset:
plot = Boxplot(dataset, other_dataset)
plot.title = "Additional dataset"
plot.execute(save=False, show=True)
