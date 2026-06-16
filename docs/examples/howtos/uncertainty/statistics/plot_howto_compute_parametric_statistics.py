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
"""# Compute parametric statistics from a dataset

## Problem

You have observed data for several variables
and want to estimate their statistics analytically
by fitting probability distributions to the data,
rather than relying on direct sample estimators.

## Solution

GEMSEO provides two parametric statistics classes backed by different libraries:

- [OTParametricStatistics][gemseo.uncertainty.statistics.ot_parametric.OTParametricStatistics]
  fits [OpenTURNS](https://openturns.github.io/openturns/latest/) distributions,
  identified by PascalCase names (e.g. `"Normal"`, `"Exponential"`, `"Uniform"`),
  with criteria such as `"BIC"` and `"Kolmogorov"`.
- [SPParametricStatistics][gemseo.uncertainty.statistics.sp_parametric.SPParametricStatistics]
  fits [SciPy](https://scipy.org/) distributions,
  identified by lowercase names (e.g. `"norm"`, `"expon"`, `"uniform"`),
  with significance-test criteria only (e.g. `"KolmogorovSmirnov"`, `"AndersonDarling"`).

Both expose the same interface for computing statistics,
tolerance intervals, and B-values.

## Step-by-step guide
"""

from __future__ import annotations

from numpy import vstack
from numpy.random import default_rng

from gemseo import create_dataset
from gemseo.uncertainty.statistics.ot_parametric import OTParametricStatistics
from gemseo.uncertainty.statistics.sp_parametric import SPParametricStatistics

# %%
# ### 1. Create synthetic data
#
# Generate 500 samples of four variables with known distributions:
# uniform, normal, Weibull, and exponential.
rng = default_rng(0)
n_samples = 500

data = vstack((
    rng.uniform(size=n_samples),
    rng.normal(size=n_samples),
    rng.weibull(1.5, size=n_samples),
    rng.exponential(size=n_samples),
)).T

variables = ["x_0", "x_1", "x_2", "x_3"]
dataset = create_dataset("Dataset", data, variables)

# %%
# ### 2. Create the parametric statistics objects
#
# Specify the candidate distributions and the fitting criterion.
#
# #### With OpenTURNS
#
# Use `"Kolmogorov"` instead of the default `"BIC"`:
ot_analysis = OTParametricStatistics(
    dataset, ["Exponential", "Normal", "Uniform"], fitting_criterion="Kolmogorov"
)
ot_analysis

# %%
# #### With SciPy
#
# The equivalent criterion is `"KolmogorovSmirnov"`:
sp_analysis = SPParametricStatistics(
    dataset, ["expon", "norm", "uniform"], fitting_criterion="KolmogorovSmirnov"
)
sp_analysis

# %%
# ### 3. Inspect the fitting matrix
#
# #### With OpenTURNS
#
# Print goodness-of-fit measures for every <variable, distribution> pair
# and see which distribution was selected for each variable:
print(sp_analysis.get_fitting_matrix())

# %%
# Plot the tested distributions over the data histogram for one variable:
ot_analysis.plot_criteria("x_0")

# %%
# #### With SciPy:
print(sp_analysis.get_fitting_matrix())

# %%
sp_analysis.plot_criteria("x_0")

# %%
# ### 4. Plot the fitted distributions
#
# CDF and PDF of the fitted distribution for each variable:
ot_analysis.plot()

# %%
# ## Summary
#
# GEMSEO wraps two backends behind the same interface:
#
# | Feature                    | OpenTURNS (`OT`)        | SciPy (`SP`)                                                           |
# |----------------------------|-------------------------|------------------------------------------------------------------------|
# | Class                      | `OTParametricStatistics`| `SPParametricStatistics`                                               |
# | Distribution names         | PascalCase (`"Normal"`) | lowercase (`"norm"`)                                                   |
# | Non-significance criteria  | BIC (default)           | —                                                                      |
# | Significance-test criteria | ChiSquared, Kolmogorov  | AndersonDarling (default), CramerVonMises, Filliben, KolmogorovSmirnov |
#
# - Both classes fit candidate distributions to each variable in a
#   [Dataset][gemseo.datasets.dataset.Dataset]
#   and select the best one per variable;
# - [get_fitting_matrix()][gemseo.uncertainty.statistics.ot_parametric.OTParametricStatistics.get_fitting_matrix]
#   shows goodness-of-fit measures and the selected distribution;
# - [plot_criteria()][gemseo.uncertainty.statistics.ot_parametric.OTParametricStatistics.plot_criteria]
#   overlays the candidate PDFs on the data histogram;
# - [plot()][gemseo.uncertainty.statistics.ot_parametric.OTParametricStatistics.plot]
#   renders the CDF and PDF of each fitted distribution.
#
# ## One step further
#
# For the full statistics catalog shared with empirical statistics,
# see [Compute statistics from a statistics object][].
