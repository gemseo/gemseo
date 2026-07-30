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
"""# Fit a probability distribution from data

## Problem

You have observed data for an uncertain variable
and want to find the probability distribution that best fits it,
so you can use that distribution in a subsequent analysis.

## Solution

GEMSEO provides two distribution fitters backed by different libraries:

- [OTDistributionFitter][gemseo.uncertainty.distribution.openturns.distribution_fitter.OTDistributionFitter]
  fits [OpenTURNS](https://openturns.github.io/openturns/latest/) distributions,
  identified by PascalCase names (e.g. `"Normal"`, `"Exponential"`),
  and offers both information-criterion (e.g. BIC) and significance-test criteria,
- [SPDistributionFitter][gemseo.uncertainty.distribution.scipy.distribution_fitter.SPDistributionFitter]
  fits [SciPy](https://scipy.org/) distributions,
  identified by lowercase names (e.g. `"norm"`, `"expon"`),
  and offers significance-test criteria only
  (e.g. `"KolmogorovSmirnov"`, `"AndersonDarling"`).

Both share the same interface:
[fit()][gemseo.uncertainty.distribution.openturns.distribution_fitter.OTDistributionFitter.fit],
[compute_measure()][gemseo.uncertainty.distribution.openturns.distribution_fitter.OTDistributionFitter.compute_measure],
and
[select()][gemseo.uncertainty.distribution.openturns.distribution_fitter.OTDistributionFitter.select].

## Step-by-step guide
"""

from __future__ import annotations

from numpy.random import default_rng

from gemseo.uncertainty.distribution.openturns.distribution_fitter import (
    OTDistributionFitter,
)
from gemseo.uncertainty.distribution.scipy.distribution_fitter import (
    SPDistributionFitter,
)

# %%
# ### 1. Prepare data
#
# For illustration, generate 100 samples from a standard normal distribution:
rng = default_rng(1)
data = rng.normal(size=100)

# %%
# ### 2. Create the distribution fitters
#
# #### With OpenTURNS:
ot_fitter = OTDistributionFitter(data)

# %%
# #### With SciPy:
sp_fitter = SPDistributionFitter(data)

# %%
# ### 3. List available distributions
#
# These are the names of the backend classes,
# e.g. `Normal` for OpenTURNS and `norm` for SciPy in the case of a normal distribution.
#
# #### With OpenTURNS (PascalCase names):
ot_fitter.available_distributions

# %%
# #### With SciPy (lowercase names):
sp_fitter.available_distributions

# %%
# ### 4. Fit individual distributions
#
# #### With OpenTURNS
#
# Fit a normal distribution — returns an
# [OTDistribution][gemseo.uncertainty.distribution.openturns.distribution.OTDistribution]:
ot_normal = ot_fitter.fit("Normal")
ot_normal

# %%
# Fit an exponential distribution:
ot_exponential = ot_fitter.fit("Exponential")
ot_exponential

# %%
# Plot the fitted distribution (PDF and CDF):
ot_normal.plot()

# %%
# #### With SciPy
#
# Fit a normal distribution — returns an
# [SPDistribution][gemseo.uncertainty.distribution.scipy.distribution.SPDistribution]:
sp_normal = sp_fitter.fit("norm")
sp_normal

# %%
# Fit an exponential distribution:
sp_exponential = sp_fitter.fit("expon")
sp_exponential

# %%
# Plot the fitted distribution:
sp_normal.plot()

# %%
# ### 5. Measure goodness-of-fit
#
# #### With OpenTURNS
#
# List available fitting criteria:
ot_fitter.available_criteria

# %%
# List criteria based on significance tests:
ot_fitter.available_significance_tests

# %%
# Compare the two distributions using the
# [Bayesian information criterion (BIC)](https://en.wikipedia.org/wiki/Bayesian_information_criterion)
# (lower is better):
ot_fitter.compute_measure(ot_normal, "BIC")

# %%
ot_fitter.compute_measure(ot_exponential, "BIC")

# %%
# Use the Kolmogorov significance test
# (returns a boolean indicating acceptability and a details dictionary):
acceptable, details = ot_fitter.compute_measure(ot_normal, "Kolmogorov")
acceptable, details

# %%
acceptable, details = ot_fitter.compute_measure(ot_exponential, "Kolmogorov")
acceptable, details

# %%
# !!! note
#
#     The significance level defaults to 0.05.
#     Pass `level=` to
#     [compute_measure()][gemseo.uncertainty.distribution.openturns.distribution_fitter.OTDistributionFitter.compute_measure]
#     to change it.

# %%
# #### With SciPy
#
# List available fitting criteria (all are significance tests):
sp_fitter.available_criteria

# %%
# Use the Kolmogorov-Smirnov significance test:
acceptable, details = sp_fitter.compute_measure(sp_normal, "KolmogorovSmirnov")
acceptable, details

# %%
acceptable, details = sp_fitter.compute_measure(sp_exponential, "KolmogorovSmirnov")
acceptable, details

# %%
# Use the Anderson-Darling significance test:
acceptable, details = sp_fitter.compute_measure(sp_normal, "AndersonDarling")
acceptable, details

# %%
# ### 6. Select the optimal distribution
#
# #### With OpenTURNS
#
# Let [select()][gemseo.uncertainty.distribution.openturns.distribution_fitter.OTDistributionFitter.select]
# automatically pick the best distribution from a list of candidates
# (`"best"` picks the one that minimizes/maximizes the criterion,
# `"first"` picks the first acceptable one):
ot_best = ot_fitter.select(["Exponential", "Normal"], "Kolmogorov")
ot_best

# %%
# #### With SciPy:
sp_best = sp_fitter.select(["expon", "norm"], "KolmogorovSmirnov")
sp_best

# %%
# ## Summary
#
# GEMSEO wraps two backends behind the same interface:
#
# | Feature                    | OpenTURNS (`OT`)        | SciPy (`SP`)                                                 |
# |----------------------------|-------------------------|--------------------------------------------------------------|
# | Class                      | `OTDistributionFitter`  | `SPDistributionFitter`                                       |
# | Distribution names         | PascalCase (`"Normal"`) | lowercase (`"norm"`)                                         |
# | Non-significance criteria  | BIC                     | —                                                            |
# | Significance-test criteria | ChiSquared, Kolmogorov  | AndersonDarling, CramerVonMises, Filliben, KolmogorovSmirnov |
#
# - [OTDistributionFitter][gemseo.uncertainty.distribution.openturns.distribution_fitter.OTDistributionFitter]
#   and
#   [SPDistributionFitter][gemseo.uncertainty.distribution.scipy.distribution_fitter.SPDistributionFitter]
#   share the same interface;
# - [fit()][gemseo.uncertainty.distribution.openturns.distribution_fitter.OTDistributionFitter.fit]
#   returns a fitted distribution object;
# - [compute_measure()][gemseo.uncertainty.distribution.openturns.distribution_fitter.OTDistributionFitter.compute_measure]
#   evaluates goodness-of-fit; significance tests return `(acceptable, details)`;
# - [select()][gemseo.uncertainty.distribution.openturns.distribution_fitter.OTDistributionFitter.select]
#   returns the best distribution from a list of candidates.
