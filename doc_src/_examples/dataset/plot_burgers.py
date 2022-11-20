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
#        :author: Syver Doving Agdestein
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Burgers dataset
===============

Dataset consisting of solutions to Burgers' equation.
"""
from __future__ import annotations

from gemseo.api import configure_logger
from gemseo.api import load_dataset
from gemseo.post.dataset.curves import Curves

configure_logger()

##############################################################################
# Load Burgers' dataset
# -----------------------
# We can easily load this dataset by means of the
# :meth:`~gemseo.api.load_dataset` function of the API:


dataset = load_dataset("BurgersDataset")
print(dataset)

##############################################################################
# Show the input and output data
# ------------------------------
print(dataset.get_data_by_group("inputs"))
print(dataset.get_data_by_group("outputs"))

##############################################################################
# Load customized dataset
# -----------------------
# Load the data with custom parameters and input-output naming.
dataset = load_dataset("BurgersDataset", n_samples=20, n_x=700, fluid_viscosity=0.03)
print(dataset)

##############################################################################
# Plot the data
# -------------
Curves(dataset, "x", "u_t").execute(save=False, show=True)
