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
"""# Compute FORM sensitivity indices

## Problem

You want to quantify how much each uncertain input variable contributes
to an *event*, e.g. a discipline output crossing a threshold,
rather than to the variability of the raw output.

## Solution

[FORMAnalysis][gemseo.uncertainty.sensitivity.form.FORMAnalysis]
turns the *importance factors*
of the first-order or second-order reliability method (FORM/SORM)
into sensitivity indices with respect to the events:

1. [compute_samples()][gemseo.uncertainty.sensitivity.form.FORMAnalysis.compute_samples]
   runs a FORM study for the events of interest;
2. [compute_indices()][gemseo.uncertainty.sensitivity.form.FORMAnalysis.compute_indices]
   derives the importance factors from the study.

!!! warning
    The outputs of interest handled by the inherited API
    (e.g. the keys of `indices` and the `output` argument of `plot()`)
    are not the disciplinary outputs but the events.

Three types of importance factors are available
(`classical`, `elliptical` and `physical`),
the `classical` ones being the default
[main_method][gemseo.uncertainty.sensitivity.base.BaseGenericSensitivityAnalysis.main_method].

## Step-by-step guide
"""

from __future__ import annotations

from gemseo.problems.uncertainty.wing_weight.discipline import WingWeightDiscipline
from gemseo.problems.uncertainty.wing_weight.uncertain_space import (
    WingWeightUncertainSpace,
)
from gemseo.uncertainty.sensitivity.form import FORMAnalysis

# %%
# ### 1. Set up the test problem
#
# The wing weight problem estimates the wing weight $W_w$ (lb) of a light aircraft
# from 10 structural and aerodynamic parameters,
# all modelled as independent uniform random variables:
discipline = WingWeightDiscipline()
uncertain_space = WingWeightUncertainSpace(
    WingWeightUncertainSpace.UniformDistribution.OPENTURNS
)

# %%
# ### 2. Instantiate a FORM analysis
analysis = FORMAnalysis()

# %%
# ### 3. Define the events of interest
#
# Build an event variable bound to the disciplinary output `Ww`,
# then the events as a mapping from names to comparisons.
# Here the failure event named `"too_heavy"` is that the wing weight exceeds 400 lb:
Ww = analysis.get_event_variables("Ww")
events = {"too_heavy": Ww > 400}

# %%
# ### 4. Run the FORM study
#
# [compute_samples()][gemseo.uncertainty.sensitivity.form.FORMAnalysis.compute_samples]
# runs the first-order reliability method (FORM) by default.
# Unlike a sampling-based sensitivity analysis,
# FORM/SORM is an optimization method searching the standard normal space
# for the most probable failure point (MPFP),
# so the returned dataset contains the model evaluations performed during this
# optimization (the optimizer iterates),
# not points drawn from a sampling of the uncertain space:
analysis.compute_samples([discipline], uncertain_space, events)

# %%
# !!! note
#     Set `algo_settings` to `OT_FORM_Settings()` or `OT_SORM_Settings()`
#     to change the settings of the reliability analysis algorithm.
#
# ### 5. Compute indices
#
# [compute_indices()][gemseo.uncertainty.sensitivity.form.FORMAnalysis.compute_indices]
# derives the importance factors from the FORM study:
analysis.compute_indices()

# %%
# ### 6. Inspect the indices
#
# The [main_indices][gemseo.uncertainty.sensitivity.form.FORMAnalysis.main_indices]
# attribute holds the classical importance factors,
# as a mapping `{event_name: [{input_name: value}]}`:
analysis.main_indices

# %%
# The three types of importance factors are available in
# [indices][gemseo.uncertainty.sensitivity.form.FORMAnalysis.indices],
# e.g. the physical ones:
analysis.indices.physical

# %%
# The underlying FORM results are stored in the dataset
# and give access to the reliability index and the failure probability:
result = analysis.dataset.misc["execution_result"]["too_heavy"]
result.probability

# %%
# ### 7. Visualize the indices
#
# A bar plot ranks the inputs by their importance for the event.
# The event name `"too_heavy"` is passed as the output of interest:
analysis.plot("too_heavy", save=False, show=True)

# %%
# ## Summary
#
# - [FORMAnalysis][gemseo.uncertainty.sensitivity.form.FORMAnalysis]
#   computes sensitivity indices from FORM/SORM importance factors;
# - [get_event_variables()][gemseo.uncertainty.sensitivity.form.FORMAnalysis.get_event_variables]
#   creates symbolic variables bound to disciplinary outputs;
# - [compute_samples()][gemseo.uncertainty.sensitivity.form.FORMAnalysis.compute_samples]
#   runs the reliability study for the events;
# - [compute_indices()][gemseo.uncertainty.sensitivity.form.FORMAnalysis.compute_indices]
#   derives the importance factors;
# - the indices are accessed via
#   [main_indices][gemseo.uncertainty.sensitivity.form.FORMAnalysis.main_indices]
#   and [indices][gemseo.uncertainty.sensitivity.form.FORMAnalysis.indices].
