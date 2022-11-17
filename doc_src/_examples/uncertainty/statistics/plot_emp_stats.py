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
Empirical estimation of statistics
==================================

In this example,
we want to empirically estimate statistics
associated with the range of the Mission discipline of the Sobieski's SSBJ problem.

For simplification,
we use uniform distributions for the discipline inputs
based on the bounds of the design parameters.
"""
from __future__ import annotations

from gemseo.api import configure_logger
from gemseo.api import create_discipline
from gemseo.api import create_scenario
from gemseo.problems.sobieski.core.problem import SobieskiProblem
from gemseo.uncertainty.api import create_statistics

configure_logger()

###############################################################################
# Create the dataset
# ------------------
# First of all,
# we create the dataset.
# For that,
# we instantiate
# the discipline :class:`~gems.problems.sobieski.disciplines.SobieskiMission`
# of the Sobieski's SSBJ problem which is known to |g|.
discipline = create_discipline("SobieskiMission")

###############################################################################
# Then,
# we load the design space of the Sobieski's SSBJ problem
# by means of the property :meth:`.SobieskiProblem.design_space`
# and :meth:`.DesignSpace.filter` the inputs of the
# discipline :class:`~gems.problems.sobieski.disciplines.SobieskiMission`.
parameter_space = SobieskiProblem().design_space
parameter_space.filter(discipline.get_input_data_names())

###############################################################################
# Then,
# we sample the discipline over this design space
# by means of a :class:`.DOEScenario`
# executed with a Monte Carlo algorithm and 100 samples.
scenario = create_scenario(
    [discipline], "DisciplinaryOpt", "y_4", parameter_space, scenario_type="DOE"
)
scenario.execute({"algo": "OT_MONTE_CARLO", "n_samples": 100})

###############################################################################
# Create an :class:`.EmpiricalStatistics` object for all variables
# ----------------------------------------------------------------
# In this second stage,
# we create an :class:`.EmpiricalStatistics`
# from the database encapsulated in a :class:`.Dataset`:
dataset = scenario.export_to_dataset(opt_naming=False)
analysis = create_statistics(dataset, name="SobieskiMission")

print(analysis)

###############################################################################
# and easily obtain statistics,
# such as the minimum values of the different variables over the dataset:
print(analysis.compute_minimum())

###############################################################################
# Create an :class:`.EmpiricalStatistics` object for the range
# ------------------------------------------------------------
# We can only reduce the statistical analysis to the range variable:
analysis = create_statistics(
    dataset, variables_names=["y_4"], name="SobieskiMission.range"
)
print(analysis)

###############################################################################
# Get minimum
# ~~~~~~~~~~~
# Here is the minimum value:
print(analysis.compute_minimum())

###############################################################################
# Get maximum
# ~~~~~~~~~~~
# Here is the maximum value:
print(analysis.compute_maximum())

###############################################################################
# Get range
# ~~~~~~~~~
# Here is the (different between minimum and maximum values):
print(analysis.compute_range())

###############################################################################
# Get mean
# ~~~~~~~~
# Here is the mean value:
print(analysis.compute_mean())

###############################################################################
# Get central moment
# ~~~~~~~~~~~~~~~~~~
# Here is the second central moment:
print(analysis.compute_moment(2))

###############################################################################
# Get standard deviation
# ~~~~~~~~~~~~~~~~~~~~~~
# Here is the standard deviation:
print(analysis.compute_standard_deviation())

###############################################################################
# Get variance
# ~~~~~~~~~~~~
# Here is the variance.
print(analysis.compute_variance())

###############################################################################
# Get quantile
# ~~~~~~~~~~~~
# Here is the quantile with level equal to 80%:
print(analysis.compute_quantile(0.8))

###############################################################################
# Get probability
# ~~~~~~~~~~~~~~~
# Here are the probability
# to respectively be greater and lower than the default output value:
default_output = discipline.execute()
print(analysis.compute_probability(default_output))
print(analysis.compute_probability(default_output, greater=False))

###############################################################################
# Get quartile
# ~~~~~~~~~~~~
# Here is the second quartile:
print(analysis.compute_quartile(2))

###############################################################################
# Get percentile
# ~~~~~~~~~~~~~~
# Here is the 50the percentile:
print(analysis.compute_percentile(50))

###############################################################################
# Get median
# ~~~~~~~~~~
# Here is the median:
print(analysis.compute_median())
