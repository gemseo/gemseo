# -*- coding: utf-8 -*-
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
Plot - Scatter matrix
=====================

"""
from __future__ import division, unicode_literals

from matplotlib import pyplot as plt

from gemseo.api import configure_logger, load_dataset

configure_logger()


############################################################################
# Load a dataset
# --------------
iris = load_dataset("IrisDataset")

############################################################################
# Plot scatter matrix
# -------------------
# We can use the :class:`.ScatterMatrix` plot where each non-diagonal block
# represents the samples according to the x- and y- coordinates names
# while the diagonal ones approximate the probability distributions of the
# variables, using either an histogram or a kernel-density estimator.
iris.plot("ScatterMatrix", classifier="specy", kde=True, show=False)
# Workaround for HTML rendering, instead of ``show=True``
plt.show()
