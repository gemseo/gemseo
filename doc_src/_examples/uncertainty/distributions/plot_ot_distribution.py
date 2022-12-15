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

from gemseo.api import configure_logger
from gemseo.uncertainty.api import create_distribution
from gemseo.uncertainty.api import get_available_distributions

configure_logger()

###############################################################################
# First of all,
# we can access the names of the available probability distributions from the API:
all_distributions = get_available_distributions()
print(all_distributions)

###############################################################################
# and filter the ones based on the OpenTURNS library
# (their names start with the acronym 'OT'):
ot_distributions = [dist for dist in all_distributions if dist.startswith("OT")]
print(ot_distributions)

###############################################################################
# Create a distribution
# ---------------------
# Then,
# we can create a probability distribution for a two-dimensional random variable
# whose components are independent and distributed
# as the standard normal distribution (mean = 0 and standard deviation = 1):
distribution_0_1 = create_distribution("x", "OTNormalDistribution", 2)
print(distribution_0_1)

###############################################################################
# or create another distribution with mean = 1 and standard deviation = 2
# for the marginal distributions:
distribution_1_2 = create_distribution(
    "x", "OTNormalDistribution", 2, mu=1.0, sigma=2.0
)
print(distribution_1_2)

###############################################################################
# We could also use the generic :class:`.OTDistribution`
# which allows access to all the OpenTURNS distributions
# but this requires to know the signature of the methods of this library:
distribution_1_2 = create_distribution(
    "x", "OTDistribution", 2, interfaced_distribution="Normal", parameters=(1.0, 2.0)
)
print(distribution_1_2)

###############################################################################
# Plot the distribution
# ---------------------
# We can plot both cumulative and probability density functions
# for the first marginal:
distribution_0_1.plot()

###############################################################################
# .. note::
#
#    We can provide a marginal index
#    as first argument of the :meth:`.Distribution.plot` method
#    but in the current version of |g|,
#    all components have the same distributions and so the plot will be the same.

###############################################################################
# Get mean
# --------
# We can access the mean of the distribution:
print(distribution_0_1.mean)

###############################################################################
# Get standard deviation
# ----------------------
# We can access the standard deviation of the distribution:
print(distribution_0_1.standard_deviation)

###############################################################################
# Get numerical range
# -------------------
# We can access the range, ie. the difference between the numerical minimum and maximum,
# of the distribution:
print(distribution_0_1.range)

###############################################################################
# Get mathematical support
# ------------------------
# We can access the range, ie. the difference between the minimum and maximum,
# of the distribution:
print(distribution_0_1.support)

###############################################################################
# Generate samples
# ----------------
# We can generate 10 samples of the distribution:
print(distribution_0_1.compute_samples(10))

###############################################################################
# Compute CDF
# -----------
# We can compute the cumulative density function component per component
# (here the probability that the first component is lower than 0.
# and that the second one is lower than 1.)::
print(distribution_0_1.compute_cdf([0.0, 1.0]))

###############################################################################
# Compute inverse CDF
# -------------------
# We can compute the inverse cumulative density function
# component per component
# (here the quantile at 50% for the first component
# and the quantile at 97.5% for the second one):
print(distribution_0_1.compute_inverse_cdf([0.5, 0.975]))
