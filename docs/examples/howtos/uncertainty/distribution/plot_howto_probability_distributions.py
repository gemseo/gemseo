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
"""# Probability distributions

## Problem

You want to model an uncertain input variable as a probability distribution,
then query its statistical properties and generate samples from it.

## Solution

GEMSEO provides two distribution backends:

- [OpenTURNS](https://openturns.github.io/openturns/latest/user_manual/probabilistic_modeling.html) (`OT`):
  classes under [gemseo.uncertainty.distributions.openturns][gemseo.uncertainty.distributions.openturns];
  named classes (e.g. [OTNormalDistribution][gemseo.uncertainty.distributions.openturns.normal.OTNormalDistribution])
  cover common distributions,
  while [OTDistribution][gemseo.uncertainty.distributions.openturns.distribution.OTDistribution]
  gives access to any OpenTURNS distribution,
- [SciPy](https://docs.scipy.org/doc/scipy/tutorial/stats/probability_distributions.html) (`SP`):
  classes under [gemseo.uncertainty.distributions.scipy][gemseo.uncertainty.distributions.scipy];
  named classes (e.g. [SPNormalDistribution][gemseo.uncertainty.distributions.scipy.normal.SPNormalDistribution])
  cover common distributions,
  while [SPDistribution][gemseo.uncertainty.distributions.scipy.distribution.SPDistribution]
  gives access to any SciPy distribution.

Both backends expose the same interface (same methods, same properties).

## Step-by-step guide
"""

from __future__ import annotations

from gemseo.uncertainty.distributions.openturns.distribution import OTDistribution
from gemseo.uncertainty.distributions.openturns.distribution_settings import (
    OTDistribution_Settings,
)
from gemseo.uncertainty.distributions.openturns.normal import OTNormalDistribution
from gemseo.uncertainty.distributions.openturns.normal_settings import (
    OTNormalDistribution_Settings,
)
from gemseo.uncertainty.distributions.scipy.distribution import SPDistribution
from gemseo.uncertainty.distributions.scipy.distribution_settings import (
    SPDistribution_Settings,
)
from gemseo.uncertainty.distributions.scipy.normal import SPNormalDistribution
from gemseo.uncertainty.distributions.scipy.normal_settings import (
    SPNormalDistribution_Settings,
)

# %%
# ### 1. Create a named distribution
#
# #### With OpenTURNS
#
# The standard normal distribution (mean = 0, standard deviation = 1):
ot_distribution_0_1 = OTNormalDistribution()
ot_distribution_0_1

# %%
# A normal distribution with custom parameters:
ot_distribution_1_2 = OTNormalDistribution(
    OTNormalDistribution_Settings(mu=1.0, sigma=2.0)
)
ot_distribution_1_2

# %%
# #### With SciPy
#
# The standard normal distribution:
sp_distribution_0_1 = SPNormalDistribution()
sp_distribution_0_1

# %%
# A normal distribution with custom parameters:
sp_distribution_1_2 = SPNormalDistribution(
    SPNormalDistribution_Settings(mu=1.0, sigma=2.0)
)
sp_distribution_1_2

# %%
# ### 2. Create a generic distribution
#
# When GEMSEO does not provide a named class for a distribution,
# use the generic class.
#
# #### With OpenTURNS
#
# Pass `interfaced_distribution` as the OpenTURNS class name
# and `parameters` as a tuple of parameter values
# (see the [OpenTURNS documentation](https://openturns.github.io/openturns/latest/user_manual/_generated/openturns.Normal.html)):
ot_distribution_generic = OTDistribution(
    OTDistribution_Settings(interfaced_distribution="Normal", parameters=(1.0, 2.0))
)
ot_distribution_generic

# %%
# #### With SciPy
#
# Pass `interfaced_distribution` as the SciPy distribution name
# and `parameters` as a dictionary of SciPy parameter names and values
# (see the [SciPy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html)):
sp_distribution_generic = SPDistribution(
    SPDistribution_Settings(
        interfaced_distribution="norm", parameters={"loc": 1.0, "scale": 2.0}
    )
)
sp_distribution_generic

# %%
# The following properties work identically for both backends;
# the examples below use the OpenTURNS distribution.
#
#
# ### 3. Plot the distribution
#
# Visualize both the probability density function (PDF) and
# the cumulative distribution function (CDF).
ot_distribution_0_1.plot()

# %%
# ### 4. Get statistics
#
# Mean:
ot_distribution_0_1.mean

# %%
# Standard deviation:
ot_distribution_0_1.standard_deviation

# %%
# Range (difference between numerical minimum and maximum):
ot_distribution_0_1.range

# %%
# Support (mathematical minimum and maximum):
ot_distribution_0_1.support

# %%
# ### 5. Evaluate the CDF and its inverse
#
# Cumulative distribution function at 0.5:
ot_distribution_0_1.compute_cdf(0.5)

# %%
# Quantile at 97.5% (inverse CDF):
ot_distribution_0_1.compute_inverse_cdf(0.975)

# %%
# ### 6. Generate samples
#
# Draw 10 samples:
ot_distribution_0_1.compute_samples(10)

# %%
# ## Summary
#
# GEMSEO wraps two backends behind the same interface:
#
# | Concept                   | OpenTURNS (`OT`)       | SciPy (`SP`)           |
# |---------------------------|------------------------|------------------------|
# | Generic class             | `OTDistribution`       | `SPDistribution`       |
# | Generic `parameters` type | `tuple`                | `dict`                 |
# | Named class               | `OTNormalDistribution` | `SPNormalDistribution` |
#
# - Named classes and their `<ClassName>_Settings` counterparts are importable from
#   [gemseo.settings.probability_distributions][gemseo.settings.probability_distributions];
# - [plot()][gemseo.uncertainty.distributions.openturns.distribution.OTDistribution.plot]
#   renders the PDF and CDF;
# - There are attributes for accessing analytical statistics.
# - [compute_samples()][gemseo.uncertainty.distributions.openturns.distribution.OTDistribution.compute_samples]
#   generates random samples.
