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
"""# Compute statistics from a statistics object

## Problem

You have an
[EmpiricalStatistics][gemseo.uncertainty.statistics.empirical.EmpiricalStatistics]
or a
[OTParametricStatistics][gemseo.uncertainty.statistics.ot_parametric.OTParametricStatistics] /
[SPParametricStatistics][gemseo.uncertainty.statistics.sp_parametric.SPParametricStatistics]
object and want to know what statistics you can compute from it.

## Solution

Both classes inherit from
[BaseStatistics][gemseo.uncertainty.statistics.base.BaseStatistics]
and share the same `compute_*` interface.
Any method documented here works identically regardless of which class you use.

## Step-by-step guide
"""

from __future__ import annotations

from gemseo import sample_disciplines
from gemseo.problems.uncertainty.wing_weight.discipline import WingWeightDiscipline
from gemseo.problems.uncertainty.wing_weight.uncertain_space import (
    WingWeightUncertainSpace,
)
from gemseo.uncertainty.statistics.empirical import EmpiricalStatistics

# %%
# ### 1. Create a statistics object
#
# The examples below use
# [EmpiricalStatistics][gemseo.uncertainty.statistics.empirical.EmpiricalStatistics],
# but the same methods are available on
# [OTParametricStatistics][gemseo.uncertainty.statistics.ot_parametric.OTParametricStatistics]
# and
# [SPParametricStatistics][gemseo.uncertainty.statistics.sp_parametric.SPParametricStatistics].
#
# Sample the
# [WingWeightDiscipline][gemseo.problems.uncertainty.wing_weight.discipline.WingWeightDiscipline]
# and restrict the analysis to the wing weight output `Ww`:
discipline = WingWeightDiscipline()
dataset = sample_disciplines(
    [discipline],
    WingWeightUncertainSpace(),
    "Ww",
    formulation_name="DisciplinaryOpt",
    algo_name="OT_MONTE_CARLO",
    n_samples=100,
)
analysis = EmpiricalStatistics(dataset, variable_names=["Ww"], name="WingWeight")
analysis

# %%
# ### 2. Compute location and spread statistics
#
# All methods return a dict mapping each variable name to its statistic value.
#
# Minimum:
analysis.compute_minimum()

# %%
# Maximum:
analysis.compute_maximum()

# %%
# Range (max - min):
analysis.compute_range()

# %%
# Mean:
analysis.compute_mean()

# %%
# Second central moment:
analysis.compute_moment(2)

# %%
# Standard deviation:
analysis.compute_standard_deviation()

# %%
# Variance:
analysis.compute_variance()

# %%
# ### 3. Compute quantile-based statistics
#
# Quantile at the 80% level:
analysis.compute_quantile(0.8)

# %%
# Second quartile (median):
analysis.compute_quartile(2)

# %%
# 50th percentile:
analysis.compute_percentile(50)

# %%
# Median:
analysis.compute_median()

# %%
# ### 4. Compute tolerance intervals
#
# Two-sided tolerance interval with 50% coverage and 95% confidence:
analysis.compute_tolerance_interval(0.5)

# %%
# B-value: left-sided tolerance interval with 90% coverage and 95% confidence:
analysis.compute_b_value()

# %%
# ### 5. Compute exceedance probability
#
# Probability that `Ww` exceeds (or falls below) its nominal value:
default_output = discipline.execute()
(
    analysis.compute_probability(default_output),
    analysis.compute_probability(default_output, greater=False),
)

# %%
# ## Summary
#
# - All methods return a dict mapping variable names to statistic values;
# - location/spread:
#   [compute_minimum()][gemseo.uncertainty.statistics.base.BaseStatistics.compute_minimum],
#   [compute_maximum()][gemseo.uncertainty.statistics.base.BaseStatistics.compute_maximum],
#   [compute_range()][gemseo.uncertainty.statistics.base.BaseStatistics.compute_range],
#   [compute_mean()][gemseo.uncertainty.statistics.base.BaseStatistics.compute_mean],
#   [compute_moment(n)][gemseo.uncertainty.statistics.base.BaseStatistics.compute_moment],
#   [compute_standard_deviation()][gemseo.uncertainty.statistics.base.BaseStatistics.compute_standard_deviation],
#   [compute_variance()][gemseo.uncertainty.statistics.base.BaseStatistics.compute_variance];
# - quantile-based:
#   [compute_quantile(p)][gemseo.uncertainty.statistics.base.BaseStatistics.compute_quantile],
#   [compute_quartile(n)][gemseo.uncertainty.statistics.base.BaseStatistics.compute_quartile],
#   [compute_percentile(n)][gemseo.uncertainty.statistics.base.BaseStatistics.compute_percentile],
#   [compute_median()][gemseo.uncertainty.statistics.base.BaseStatistics.compute_median];
# - tolerance:
#   [compute_tolerance_interval(cov)][gemseo.uncertainty.statistics.base.BaseStatistics.compute_tolerance_interval],
#   [compute_b_value()][gemseo.uncertainty.statistics.base.BaseStatistics.compute_b_value];
# - probability:
#   [compute_probability(values)][gemseo.uncertainty.statistics.base.BaseStatistics.compute_probability],
#   [compute_joint_probability(values)][gemseo.uncertainty.statistics.empirical.EmpiricalStatistics.compute_joint_probability]
#   ([EmpiricalStatistics][gemseo.uncertainty.statistics.empirical.EmpiricalStatistics] only).
#
# ## One step further
#
# - [Compute empirical statistics from a dataset][]
#   shows how to build an
#   [EmpiricalStatistics][gemseo.uncertainty.statistics.empirical.EmpiricalStatistics]
#   object and plot the empirical distribution.
# - [Compute parametric statistics from a dataset][]
#   shows how to fit distributions to data,
#   build a parametric statistics object,
#   and plot the fitted distributions.
