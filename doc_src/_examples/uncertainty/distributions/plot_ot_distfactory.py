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
Fitting a distribution from data based on OpenTURNS
===================================================
"""
from __future__ import annotations

from gemseo.api import configure_logger
from gemseo.uncertainty.distributions.openturns.fitting import OTDistributionFitter
from numpy.random import randn
from numpy.random import seed

configure_logger()

######################################################################################
# In this example,
# we will see how to fit a distribution from data.
# For a purely pedagogical reason,
# we consider a synthetic dataset made of 100 realizations of *'X'*,
# a random variable distributed according to the standard normal distribution.
# These samples are generated from the NumPy library.
seed(1)
data = randn(100)
variable_name = "X"

###############################################################################
# Create a distribution fitter
# ----------------------------
# Then,
# we create an :class:`.OTDistributionFitter` from these data and this variable name:
fitter = OTDistributionFitter(variable_name, data)

###############################################################################
# Fit a distribution
# ------------------
# From this distribution fitter,
# we can easily fit any distribution available in the OpenTURNS library:
print(fitter.available_distributions)

###############################################################################
# For example,
# we can fit a normal distribution:
norm_dist = fitter.fit("Normal")
print(norm_dist)

###############################################################################
# or an exponential one:
exp_dist = fitter.fit("Exponential")
print(exp_dist)

###############################################################################
# The returned object is an :class:`.OTDistribution`
# that we can represent graphically
# in terms of probability and cumulative density functions:
norm_dist.plot()

###############################################################################
# Measure the goodness-of-fit
# ---------------------------
# We can also measure the goodness-of-fit of a distribution
# by means of a fitting criterion.
# Some fitting criteria are based on significance tests
# made of a test statistics, a p-value and a significance level.
# We can access the names of the available fitting criteria:
print(fitter.available_criteria)
print(fitter.available_significance_tests)

###############################################################################
# For example,
# we can measure the goodness-of-fit of the previous distributions
# by considering the `Bayesian information criterion (BIC)
# <https://en.wikipedia.org/wiki/Bayesian_information_criterion>`_:
quality_measure = fitter.compute_measure(norm_dist, "BIC")
print("Normal: ", quality_measure)

quality_measure = fitter.compute_measure(exp_dist, "BIC")
print("Exponential: ", quality_measure)

###############################################################################
# Here,
# the fitted normal distribution is better than the fitted exponential one
# in terms of BIC.
# We can also the Kolmogorov fitting criterion
# which is based on the Kolmogorov significance test:
acceptable, details = fitter.compute_measure(norm_dist, "Kolmogorov")
print("Normal: ", acceptable, details)
acceptable, details = fitter.compute_measure(exp_dist, "Kolmogorov")
print("Exponential: ", acceptable, details)

###############################################################################
# In this case,
# the :meth:`.OTDistributionFitter.measure` method returns a tuple with two values:
#
# 1. a boolean
#    indicating if the measured distribution is acceptable to model the data,
# 2. a dictionary containing the test statistics,
#    the p-value and the significance level.
#
# .. note::
#     We can also change the significance level for significance tests
#     whose default value is 0.05.
#     For that, use the :code:`level` argument.

###############################################################################
# Select an optimal distribution
# ------------------------------
# Lastly,
# we can also select an optimal :class:`.OTDistribution`
# based on a collection of distributions names,
# a fitting criterion,
# a significance level
# and a selection criterion:
#
# - 'best': select the distribution
#   minimizing (or maximizing, depending on the criterion) the criterion,
# - 'first': select the first distribution
#   for which the criterion is greater (or lower, depending on the criterion)
#   than the level.
#
# By default,
# the :meth:`.OTDistributionFitter.select` method uses a significance level equal to 0.5
# and 'best' selection criterion.
selected_distribution = fitter.select(["Exponential", "Normal"], "Kolmogorov")
print(selected_distribution)
