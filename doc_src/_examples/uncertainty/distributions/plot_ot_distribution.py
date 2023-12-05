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
Probability distributions based on OpenTURNS
============================================

In this example,
we seek to create a probability distribution based on the OpenTURNS library.
"""

from __future__ import annotations

from gemseo import configure_logger
from gemseo.uncertainty import create_distribution
from gemseo.uncertainty import get_available_distributions

configure_logger()

# %%
# First of all,
# we can access the names of the available probability distributions from the API:
all_distributions = get_available_distributions()
all_distributions

# %%
# and filter the ones based on the OpenTURNS library
# (their names start with the acronym 'OT'):
ot_distributions = get_available_distributions("OTDistribution")
ot_distributions

# %%
# Create a distribution
# ---------------------
# Then,
# we can create a probability distribution for a two-dimensional random variable
# with independent components that follow a normal distribution.
#
# Case 1: the OpenTURNS distribution has a GEMSEO class
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# For the standard normal distribution (mean = 0 and standard deviation = 1):
distribution_0_1 = create_distribution("x", "OTNormalDistribution", 2)
distribution_0_1

# %%
# For a normal with mean = 1 and standard deviation = 2:
distribution_1_2 = create_distribution(
    "x", "OTNormalDistribution", 2, mu=1.0, sigma=2.0
)
distribution_1_2

# %%
# Case 2: the OpenTURNS distribution has no GEMSEO class
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# When GEMSEO does not offer a class for the OpenTURNS distribution,
# we can use the generic GEMSEO class :class:`.OTDistribution`
# to create any OpenTURNS distribution
# by setting ``interfaced_distribution`` to its OpenTURNS name
# and ``parameters`` as a tuple of OpenTURNS parameter values
# (`see the documentation of OpenTURNS
# <https://openturns.github.io/openturns/latest/user_manual/_generated/
# openturns.Normal.html#openturns.Normal>`__).
distribution_1_2 = create_distribution(
    "x", "OTDistribution", 2, interfaced_distribution="Normal", parameters=(1.0, 2.0)
)
distribution_1_2

# %%
# Plot the distribution
# ---------------------
# We can plot both cumulative and probability density functions
# for the first marginal:
distribution_0_1.plot()

# %%
# .. note::
#
#    We can provide a marginal index
#    as first argument of the :meth:`.Distribution.plot` method
#    but in the current version of |g|,
#    all components have the same distributions and so the plot will be the same.

# %%
# Get statistics
# --------------
# Mean
# ~~~~
# We can access the mean of the distribution:
distribution_0_1.mean

# %%
# Standard deviation
# ~~~~~~~~~~~~~~~~~~
# We can access the standard deviation of the distribution:
distribution_0_1.standard_deviation

# %%
# Numerical range
# ~~~~~~~~~~~~~~~
# We can access the range,
# i.e. the difference between the numerical minimum and maximum,
# of the distribution:
distribution_0_1.range

# %%
# Mathematical support
# ~~~~~~~~~~~~~~~~~~~~
# We can access the range,
# i.e. the difference between the minimum and maximum,
# of the distribution:
distribution_0_1.support

# %%
# Compute CDF
# -----------
# We can compute the cumulative density function component per component
# (here the probability that the first component is lower than 0.
# and that the second one is lower than 1.):
distribution_0_1.compute_cdf([0.0, 1.0])

# %%
# Compute inverse CDF
# -------------------
# We can compute the inverse cumulative density function
# component per component
# (here the quantile at 50% for the first component
# and the quantile at 97.5% for the second one):
distribution_0_1.compute_inverse_cdf([0.5, 0.975])

# %%
# Generate samples
# ----------------
# We can generate 10 samples of the distribution:
distribution_0_1.compute_samples(10)
