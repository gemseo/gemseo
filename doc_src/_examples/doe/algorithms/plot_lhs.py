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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Optimal LHS vs LHS
==================
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from numpy import linspace

from gemseo import compute_doe

# %%
# Latin hypercube sampling (LHS) is a technique
# to generate a set of :math:`n` points in dimension :math:`d`,
# with good space-filling properties.
# LHS is also the name of the resulting design of experiments (DOE).
#
# We can use the ``"OT_LHS"`` algorithm,
# coming from OpenTURNS as indicated by the ``"OT_"`` prefix,
# to generate such a DOE,
# say with 15 points in dimension 2:
n = 15
d = 2
samples = compute_doe(d, algo_name="OT_LHS", n_samples=n)

plt.plot(samples[:, 0], samples[:, 1], "o")
plt.xticks(linspace(0, 1, n + 1), minor=True)
plt.yticks(linspace(0, 1, n + 1), minor=True)
plt.grid(which="both")
plt.title("An LHS")
plt.show()

# %%
# From a technical point of view,
# the range of each variable is divided into :math:`n` equally probable intervals.
# Then,
# the :math:`n` points are added sequentially
# to satisfy the Latin hypercube requirement:
# one and only one point per interval.
# When adding a point,
# an interval is chosen at random for each variable,
# conditionally to this requirement,
# then the point is drawn uniformly into the resulting hypercube.
# Thus,
# LHS is not a deterministic technique,
# and so we can generate another LHS (if we change the ``seed``):
samples = compute_doe(d, algo_name="OT_LHS", n_samples=n, seed=123)

plt.plot(samples[:, 0], samples[:, 1], "o")
plt.xticks(linspace(0, 1, n + 1), minor=True)
plt.yticks(linspace(0, 1, n + 1), minor=True)
plt.grid(which="both")
plt.title("Another LHS")
plt.show()

# %%
# These DOEs are different,
# but share things in common:
# they cover the space well in some places
# and bad in others with points close to each other.
# For both DOEs, there is room for improvement.
# To search for this improvement,
# one can use the ``"OT_OPT_LHS"`` algorithm
# by disabling its ``annealing`` option,
# to select the best LHS among a 1000 Monte Carlo instances:
samples = compute_doe(d, algo_name="OT_OPT_LHS", n_samples=n, annealing=False)

plt.plot(samples[:, 0], samples[:, 1], "o")
plt.xticks(linspace(0, 1, n + 1), minor=True)
plt.yticks(linspace(0, 1, n + 1), minor=True)
plt.grid(which="both")
plt.title("An LHS optimized by Monte Carlo")
plt.show()

# %%
# The result is a little better but there is still room for improvement.
# Finally,
# we can use the ``"OT_OPT_LHS"`` algorithm with its default settings,
# to get an LHS improved by simulated annealing, a global optimization algorithm.
samples = compute_doe(d, algo_name="OT_OPT_LHS", n_samples=n)

plt.plot(samples[:, 0], samples[:, 1], "o")
plt.xticks(linspace(0, 1, n + 1), minor=True)
plt.yticks(linspace(0, 1, n + 1), minor=True)
plt.grid(which="both")
plt.title("An LHS optimized by simulated annealing")
plt.show()

# %%
# We see that this DOE covers the space much better.
#
# .. note::
#    These DOEs are optimal according to the C2 discrepancy,
#    measuring the distance between the empirical distribution of the points
#    and the uniform distribution.
#    ``"OT_OPT_LHS"`` has options to change
#    this space-filling criterion (``criterion``)
#    the number of Monte Carlo instances (``n_replicates``),
#    and the profile temperature for simulated annealing (``temperature``).
#
#    See :ref:`OT_OPT_LHS_options` for more information about the settings.
#
# .. seealso::
#    This example uses the ``"OT_OPT_LHS"`` algorithm from OpenTURNS
#    to create an optimal LHS.
#    For the same purpose,
#    we could also use the ``"LHS"`` algorithm from SciPy
#    with its option ``optimization`` set to ``"random-cd"`` or ``"lloyd"``.
#
#    See :ref:`LHS_options` for more information about the settings.
