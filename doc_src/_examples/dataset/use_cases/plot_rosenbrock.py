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
Rosenbrock dataset
==================

This :class:`.Dataset` contains 100 evaluations
of the well-known Rosenbrock function:

.. math::

   f(x,y)=(1-x)^2+100(y-x^2)^2

This function is known for its global minimum at point (1,1),
its banana valley and the difficulty to reach its minimum.

This :class:`.Dataset` is based on a full-factorial
design of experiments.

`More information about the Rosenbrock function
<https://en.wikipedia.org/wiki/Rosenbrock_function>`_
"""
from __future__ import annotations

from gemseo.api import configure_logger
from gemseo.api import load_dataset
from gemseo.post.dataset.yvsx import YvsX
from gemseo.post.dataset.zvsxy import ZvsXY

configure_logger()


##############################################################################
# Load Rosenbrock dataset
# -----------------------
# We can easily load this dataset by means of the
# :meth:`~gemseo.api.load_dataset` function of the API:

dataset = load_dataset("RosenbrockDataset")
print(dataset)

##############################################################################
# Show the input and output data
# ------------------------------
print(dataset.get_data_by_group("design_parameters"))
print(dataset.get_data_by_group("functions"))

##############################################################################
# Load the data with an input-output naming
# -----------------------------------------
dataset = load_dataset("RosenbrockDataset", opt_naming=False)
print(dataset)

##############################################################################
# Plot the data
# -------------
ZvsXY(dataset, x="x", x_comp=0, y="x", y_comp=1, z="rosen").execute(
    save=False, show=True
)

YvsX(dataset, x="x", x_comp=0, y="rosen").execute(save=False, show=True)
