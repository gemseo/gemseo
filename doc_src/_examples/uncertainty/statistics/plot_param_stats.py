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
Parametric estimation of statistics
===================================

In this example,
we want to estimate statistics from synthetic data.
These data are 500 realizations of x_0, x_1, x_2 and x_3
distributed in the following way:

- x_0: standard uniform distribution,
- x_1: standard normal distribution,
- x_2: standard Weibull distribution,
- x_3: standard exponential distribution.

These samples are generated from the NumPy library.
"""
from __future__ import annotations

from gemseo.api import configure_logger
from gemseo.api import create_dataset
from gemseo.uncertainty.api import create_statistics
from numpy import vstack
from numpy.random import exponential
from numpy.random import normal
from numpy.random import rand
from numpy.random import seed
from numpy.random import weibull

configure_logger()

# %%
# Create synthetic data
# ---------------------

seed(0)

n_samples = 500

uniform_rand = rand(n_samples)
normal_rand = normal(size=n_samples)
weibull_rand = weibull(1.5, size=n_samples)
exponential_rand = exponential(size=n_samples)

data = vstack((uniform_rand, normal_rand, weibull_rand, exponential_rand)).T

variables = ["x_0", "x_1", "x_2", "x_3"]

print(data)

# %%
# Create a :class:`.ParametricStatistics` object
# ----------------------------------------------
# We create a :class:`.ParametricStatistics` object
# from this data encapsulated in a :class:`.Dataset`:

dataset = create_dataset("Dataset", data, variables)

# %%
# and a list of names of candidate probability distributions:
# exponential, normal and uniform distributions
# (see :meth:`.ParametricStatistics.get_available_distributions`).
# We do not use the default
# fitting criterion ('BIC') but 'Kolmogorov'
# (see :meth:`.ParametricStatistics.get_available_criteria`
# and :meth:`.ParametricStatistics.get_significance_tests`).

tested_distributions = ["Exponential", "Normal", "Uniform"]
analysis = create_statistics(
    dataset, tested_distributions=tested_distributions, fitting_criterion="Kolmogorov"
)
print(analysis)

# %%
# Print the fitting matrix
# ------------------------
# At this step,
# an optimal distribution has been selected for each variable
# based on the tested distributions and on the Kolmogorov fitting criterion.
# We can print the fitting matrix to see
# the goodness-of-fit measures for each pair < variable, distribution >
# as well as the selected distribution for each variable.
# Note that in the case of significance tests,
# the goodness-of-fit measures are the p-values.
print(analysis.get_fitting_matrix())

# %%
# One can also plot the tested distributions over an histogram of the data
# as well as the corresponding values of the Kolmogorov fitting criterion:
analysis.plot_criteria("x_0")

# %%
# Get statistics
# --------------
# From this :class:`.ParametricStatistics` instance,
# we can easily get statistics for the different variables
# based on the selected distributions.

# %%
# Get minimum
# ~~~~~~~~~~~
# Here is the minimum value for the different variables:
print(analysis.compute_minimum())

# %%
# Get maximum
# ~~~~~~~~~~~
# Here is the minimum value for the different variables:
print(analysis.compute_maximum())

# %%
# Get range
# ~~~~~~~~~
# Here is the range,
# i.e. the difference between the minimum and maximum values,
# for the different variables:
print(analysis.compute_range())

# %%
# Get mean
# ~~~~~~~~
# Here is the mean value for the different variables:
print(analysis.compute_mean())

# %%
# Get standard deviation
# ~~~~~~~~~~~~~~~~~~~~~~
# Here is the standard deviation for the different variables:
print(analysis.compute_standard_deviation())

# %%
# Get variance
# ~~~~~~~~~~~~
# Here is the variance for the different variables:
print(analysis.compute_variance())

# %%
# Get quantile
# ~~~~~~~~~~~~
# Here is the quantile with level 80% for the different variables:
print(analysis.compute_quantile(0.8))

# %%
# Get quartile
# ~~~~~~~~~~~~
# Here is the second quartile for the different variables:
print(analysis.compute_quartile(2))

# %%
# Get percentile
# ~~~~~~~~~~~~~~
# Here is the 50th percentile for the different variables:
print(analysis.compute_percentile(50))

# %%
# Get median
# ~~~~~~~~~~
# Here is the median for the different variables:
print(analysis.compute_median())

# %%
# Get tolerance interval
# ~~~~~~~~~~~~~~~~~~~~~~
# Here is the two-sided tolerance interval with a coverage level equal to 50%
# with a confidence level of 95% for the different variables:
print(analysis.compute_tolerance_interval(0.5))

# %%
# Get B-value
# ~~~~~~~~~~~
# Here is the B-value for the different variables, which is a left-sided
# tolerance interval with a coverage level equal to 90%
# with a confidence level of 95%:
print(analysis.compute_b_value())
