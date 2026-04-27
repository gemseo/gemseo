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
"""# Improve LHS coverage

## Problem

Latin hypercube sampling (LHS) is a pseudo-random DOE technique.
Given a number of samples and a dimension,
calling an LHS algorithm twice will lead to two different sets of points.
For studies where space-fillingness matters,
e.g. surrogate modelling,
we are looking for a set that maximizes the coverage of the space.

## Solution

Use an optimized LHS algorithm instead of a standard one.
It searches for the LHS with the best space-filling properties.

## Step-by-step guide
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from numpy import linspace

from gemseo import compute_doe
from gemseo.algos.doe.openturns.settings.ot_lhs import OT_LHS_Settings
from gemseo.algos.doe.openturns.settings.ot_opt_lhs import OT_OPT_LHS_Settings

# %%
# ### 1. Generate a standard LHS
#
# Latin hypercube sampling divides the range of each variable into $n$ equally
# probable intervals and places exactly one point per interval.
# Points are drawn randomly within each interval,
# so the result changes with the seed and the coverage can be uneven:
# some regions end up dense, others sparse.
n = 15
d = 2
samples = compute_doe(d, settings_model=OT_LHS_Settings(n_samples=n))

plt.plot(samples[:, 0], samples[:, 1], "o")
plt.xticks(linspace(0, 1, n + 1), minor=True)
plt.yticks(linspace(0, 1, n + 1), minor=True)
plt.grid(which="both")
plt.title("Standard LHS")
plt.show()

# %%
# Running the algorithm again with a different seed confirms
# that the result is non-deterministic and the coverage varies:
samples = compute_doe(d, settings_model=OT_LHS_Settings(n_samples=n, seed=123))

plt.plot(samples[:, 0], samples[:, 1], "o")
plt.xticks(linspace(0, 1, n + 1), minor=True)
plt.yticks(linspace(0, 1, n + 1), minor=True)
plt.grid(which="both")
plt.title("Standard LHS with a different seed")
plt.show()

# %%
# ### 2. Improve coverage with Monte Carlo selection
#
# An optimized LHS can be called by using the `"OT_OPT_LHS"` algorithm from OpenTURNS.
# `"OT_OPT_LHS"` with `annealing=False` generates a large number of random LHS
# candidates (1000 by default) and returns the one with the best space-filling
# score, measured by the $C^2$ discrepancy — a criterion that quantifies how far
# the empirical distribution of points deviates from the uniform distribution.
# This is a simple but effective improvement over a single random LHS:
samples = compute_doe(
    d, settings_model=OT_OPT_LHS_Settings(n_samples=n, annealing=False)
)

plt.plot(samples[:, 0], samples[:, 1], "o")
plt.xticks(linspace(0, 1, n + 1), minor=True)
plt.yticks(linspace(0, 1, n + 1), minor=True)
plt.grid(which="both")
plt.title("LHS optimised by Monte Carlo")
plt.show()

# %%
# ### 3. Further improve coverage with simulated annealing
#
# `"OT_OPT_LHS"` with its default settings (`annealing=True`) goes further:
# starting from the best Monte Carlo candidate,
# it applies simulated annealing — a global optimisation algorithm that
# iteratively perturbs the design and accepts degradations with a decreasing
# probability, allowing it to escape local optima and converge to a
# near-optimal space-filling design.
# The result covers the space much more uniformly:
samples = compute_doe(d, settings_model=OT_OPT_LHS_Settings(n_samples=n))

plt.plot(samples[:, 0], samples[:, 1], "o")
plt.xticks(linspace(0, 1, n + 1), minor=True)
plt.yticks(linspace(0, 1, n + 1), minor=True)
plt.grid(which="both")
plt.title("LHS optimised by simulated annealing")
plt.show()

# %%
# ## Summary
#
# An LHS algorithm (such as `"OT_LHS"`) produces a valid
# but possibly poorly space-filling DOE.
# An optimized LHS (such as `"OT_OPT_LHS"`) searches for the best LHS according to the C2 discrepancy:
# use `annealing=False` for a fast Monte Carlo selection,
# or keep the default `annealing=True` for a near-optimal design via simulated annealing.
#
# ## One step further
#
# The optimized LHS `"OT_OPT_LHS"` exposes additional settings
# to fine-tune the optimisation:
# `criterion` to change the space-filling criterion,
# `n_replicates` to control the number of Monte Carlo candidates,
# and `temperature` to adjust the simulated annealing profile.
# SciPy's `"LHS"` algorithm offers similar optimisation capabilities
# via its `optimization` option (`"random-cd"` or `"lloyd"`).
