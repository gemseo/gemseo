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
"""# Compute empirical statistics from a dataset

## Problem

You have evaluated a discipline at many input points
and want to quantify the variability of its outputs
(mean, variance, quantiles, exceedance probabilities, …)
without assuming any parametric form for the output distribution.

## Solution

[EmpiricalStatistics][gemseo.uncertainty.statistics.empirical.EmpiricalStatistics]
estimates statistics directly from a
[Dataset][gemseo.datasets.dataset.Dataset]
using sample estimators, with no distributional assumption.
It also provides graphical tools (boxplot, CDF, PDF).

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
# ### 1. Create a dataset
#
# Sample the
# [WingWeightDiscipline][gemseo.problems.uncertainty.wing_weight.discipline.WingWeightDiscipline]
# discipline over its uncertain space using Monte Carlo sampling:
discipline = WingWeightDiscipline()
parameter_space = WingWeightUncertainSpace()
dataset = sample_disciplines(
    [discipline],
    parameter_space,
    "Ww",
    formulation_name="DisciplinaryOpt",
    algo_name="OT_MONTE_CARLO",
    n_samples=100,
)

# %%
# ### 2. Create an EmpiricalStatistics object
#
# Pass the dataset to
# [EmpiricalStatistics][gemseo.uncertainty.statistics.empirical.EmpiricalStatistics].
# By default, all variables in the dataset are included:
analysis = EmpiricalStatistics(dataset, name="WingWeightDiscipline")
analysis

# %%
# ### 3. Restrict to variables of interest
#
# Focus on the wing weight variable `Ww` only:
analysis = EmpiricalStatistics(
    dataset, variable_names=["Ww"], name="WingWeightDiscipline.weight"
)
analysis

# %%
# ### 4. Plot the empirical distribution
#
# Boxplot:
analysis.plot_boxplot()

# %%
# Empirical CDF:
analysis.plot_cdf()

# %%
# Empirical PDF:
analysis.plot_pdf()

# %%
# ## Summary
#
# - [EmpiricalStatistics][gemseo.uncertainty.statistics.empirical.EmpiricalStatistics]
#   estimates statistics from a
#   [Dataset][gemseo.datasets.dataset.Dataset] with no distributional assumption;
# - pass `variable_names` to restrict the analysis to specific variables;
# - visualization:
#   [plot_boxplot()][gemseo.uncertainty.statistics.empirical.EmpiricalStatistics.plot_boxplot],
#   [plot_cdf()][gemseo.uncertainty.statistics.empirical.EmpiricalStatistics.plot_cdf],
#   [plot_pdf()][gemseo.uncertainty.statistics.empirical.EmpiricalStatistics.plot_pdf].
#
# ## One step further
#
# For the full statistics catalog shared with parametric statistics,
# see [Compute statistics from a statistics object][].
