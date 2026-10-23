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
"""# Compute the Sobol' indices of a rare event

## Problem

You want the Sobol' indices of a binary event,
e.g. a discipline output exceeding a threshold,
rather than those of the raw output.
A crude Monte Carlo estimation of these indices is intractable
when the event is rare.

## Solution

[ROSobolAnalysis][gemseo.uncertainty.sensitivity.ro_sobol.ROSobolAnalysis]
estimates the Sobol' indices of [rare events][concept-rosobol-analysis]
by combining the first-order reliability method (FORM)
to locate the design point,
an importance sampling auxiliary density centered on this design point
and importance sampling estimators of the Sobol' indices:

1. [get_event_variables()][gemseo.uncertainty.sensitivity.ro_sobol.ROSobolAnalysis.get_event_variables]
   creates the symbolic variables used to define the events;
2. [compute_samples()][gemseo.uncertainty.sensitivity.ro_sobol.ROSobolAnalysis.compute_samples]
   locates the design point of each event and draws the samples;
3. [compute_indices()][gemseo.uncertainty.sensitivity.ro_sobol.ROSobolAnalysis.compute_indices]
   estimates the Sobol' indices from these samples.

!!! warning
    For the Sobol' indices of a raw disciplinary output,
    use [SobolAnalysis][gemseo.uncertainty.sensitivity.sobol.SobolAnalysis].
    Here, the outputs of interest handled by the inherited API
    (e.g. the keys of `indices` and the `output` argument of `plot()`)
    are not the disciplinary outputs but the events.

## Step-by-step guide
"""

from __future__ import annotations

from gemseo.problem.uncertainty.wing_weight.discipline import WingWeightDiscipline
from gemseo.problem.uncertainty.wing_weight.random_space import WingWeightRandomSpace
from gemseo.uncertainty.sensitivity.ro_sobol import ROSobolAnalysis

# %%
# ### Prerequisites
#
# This how-to needs one or more disciplines
# and a random space whose distributions come from OpenTURNS.
#
# The wing weight problem estimates the wing weight $W_w$ (lb) of a light aircraft
# from 10 structural and aerodynamic parameters,
# all modelled as independent uniform random variables,
# here with OpenTURNS distributions:
discipline = WingWeightDiscipline()
random_space = WingWeightRandomSpace(
    WingWeightRandomSpace.UniformDistribution.OPENTURNS
)

# %%
# ### 1. Instantiate a reliability-oriented Sobol' analysis
analysis = ROSobolAnalysis()

# %%
# ### 2. Define the events of interest
#
# Build an event variable bound to the disciplinary output `Ww`,
# then the events as a mapping from names to comparisons.
# Here the event named `"too_heavy"` is that the wing weight exceeds 400 lb:
Ww = analysis.get_event_variables("Ww")
events = {"too_heavy": Ww > 400.0}

# %%
# ### 3. Draw the samples
#
# [compute_samples()][gemseo.uncertainty.sensitivity.ro_sobol.ROSobolAnalysis.compute_samples]
# first runs FORM to locate the design point of the event,
# i.e. the most probable way for the event to happen:
# the point of the event domain closest to the origin
# once the inputs are mapped to independent standard normal variables
# (the standard space).
# It then draws the samples from the importance sampling auxiliary density
# and evaluates the model at these samples.
#
# The argument `n_samples` is the total budget of model evaluations,
# FORM evaluations included:
analysis.compute_samples([discipline], random_space, events, n_samples=10000)

# %%
# ### 4. Compute the indices
#
# [compute_indices()][gemseo.uncertainty.sensitivity.ro_sobol.ROSobolAnalysis.compute_indices]
# estimates the Sobol' indices of each event:
analysis.compute_indices()

# %%
# ### 5. Inspect the results
#
# The indices are stored in
# [indices][gemseo.uncertainty.sensitivity.ro_sobol.ROSobolAnalysis.indices]
# as mappings `{event_name: [{input_name: array}]}`,
# except the second-order ones,
# which are nested one level deeper:
# `{event_name: [{input_name: {input_name: array}}]}`.
# Here are the total-order indices:
analysis.indices.total["too_heavy"]

# %%
# the first-order ones:
analysis.indices.first["too_heavy"]

# %%
# the second-order ones:
analysis.indices.second["too_heavy"]

# %%
# the probability of the event estimated by importance sampling:
analysis.dataset.misc["probability"]["too_heavy"]

# %%
# the design point found by FORM, in the standard space:
design_point = analysis.dataset.misc["design_point"]["too_heavy"]
random_space.convert_array_to_dict(design_point)

# %%
# and the confidence intervals of the first-order indices with
# [get_intervals()][gemseo.uncertainty.sensitivity.ro_sobol.ROSobolAnalysis.get_intervals]:
analysis.get_intervals()

# %%
# Pass `first_order=False` for those of the total-order indices:
analysis.get_intervals(first_order=False)

# %%
# ### 6. Visualize the indices
#
# The subtitle of the plot shows the estimated probability of the event:
analysis.plot("too_heavy", save=False, show=True)

# %%
# [plot_second_order()][gemseo.uncertainty.sensitivity.ro_sobol.ROSobolAnalysis.plot_second_order]
# displays the second-order indices as a symmetric heat map:
analysis.plot_second_order("too_heavy").execute(save=False, show=True)

# %%
# ## Summary
#
# - [ROSobolAnalysis][gemseo.uncertainty.sensitivity.ro_sobol.ROSobolAnalysis]
#   estimates the Sobol' indices of rare events;
# - [get_event_variables()][gemseo.uncertainty.sensitivity.ro_sobol.ROSobolAnalysis.get_event_variables]
#   creates the symbolic variables used to define the events;
# - [compute_samples()][gemseo.uncertainty.sensitivity.ro_sobol.ROSobolAnalysis.compute_samples]
#   locates the design point of each event with FORM
#   and draws the importance sampling design within the budget `n_samples`;
# - [compute_indices()][gemseo.uncertainty.sensitivity.ro_sobol.ROSobolAnalysis.compute_indices]
#   estimates the Sobol' indices of the events;
# - the indices are accessed via
#   [indices][gemseo.uncertainty.sensitivity.ro_sobol.ROSobolAnalysis.indices]
#   and their confidence intervals via
#   [get_intervals()][gemseo.uncertainty.sensitivity.ro_sobol.ROSobolAnalysis.get_intervals],
#   for the first-order indices by default
#   and for the total-order ones with `first_order=False`;
# - [plot()][gemseo.uncertainty.sensitivity.ro_sobol.ROSobolAnalysis.plot]
#   displays the first-order and total-order indices
#   with their confidence intervals
#   and [plot_second_order()][gemseo.uncertainty.sensitivity.ro_sobol.ROSobolAnalysis.plot_second_order]
#   the second-order ones.
#
# ## One step further
#
# The first-order indices are close to zero
# while the total-order ones are large:
# the event results from the joint effect of several inputs,
# which is typical of rare events.
# `Nz`, `tc`, `A`, `Sw` and `Wdg` drive the event
# while `q` and `Wfw` have a negligible influence.
#
# The second-order indices tell
# how much of this joint effect comes from pairs of inputs.
# Each of them is below 0.05
# and the pairs involving `Nz`, `tc`, `A`, `Sw` and `Wdg` are the largest ones.
# For each of these inputs,
# the second-order indices of its pairs add up to about 0.1 to 0.2,
# far less than the gap between its total-order and first-order indices:
# most of this gap comes from interactions with two other inputs or more,
# i.e. the event occurs when several inputs are extreme at the same time.
# In general,
# a large gap between the total-order and first-order indices
# that the second-order indices do not fill
# reveals higher-order interactions.
#
# !!! warning "Small indices are imprecise"
#     Like the first-order indices,
#     the second-order ones are small,
#     so their estimation error is large relative to their values:
#     trust their order of magnitude rather than the ranking of close pairs.
#
# The largest coordinates of the design point in absolute value
# correspond to the most influential inputs.
#
# !!! note "The indices are computed in the standard space"
#     For independent marginals,
#     the standard inputs map one-to-one to the physical inputs
#     and share their names.
#
# What is left of the budget `n_samples` after FORM
# is split equally between the events.
# By default,
# the second-order indices are also computed
# and each pick-and-freeze block has $N=b/(2+2d)$ samples,
# where $b$ is the budget left for the event and $d=10$ is the number of inputs.
#
# !!! tip "Skip the second-order indices"
#     Pass `compute_second_order=False` to `compute_samples()`
#     when the second-order indices are not needed:
#     each block then has $N=b/(2+d)$ samples instead of $b/(2+2d)$,
#     which improves the accuracy of the first-order and total-order indices
#     for the same budget.
#
# !!! tip "Analyze several events"
#     Several events can be passed in `events`.
#     Each one is processed independently,
#     and all of them share the budget `n_samples`.
#
# !!! tip "Use the rank-based estimator"
#     Set `algo_settings` to any DOE settings other than
#     [OT_SOBOL_INDICES_Settings][gemseo.doe.openturns.settings.ot_sobol_indices.OT_SOBOL_INDICES_Settings],
#     e.g. `OT_MONTE_CARLO_Settings()` imported with
#     `from gemseo.doe import OT_MONTE_CARLO_Settings`,
#     to use the rank-based estimator,
#     which only provides the first-order indices
#     but uses a design whose size does not grow with the number of inputs.
#     The content of these settings is ignored:
#     the samples are drawn independently from the auxiliary density
#     within the budget `n_samples` and with the seed `seed`
#     of `compute_samples()`.
#
# !!! tip "Tune the FORM stage"
#     Set `form_settings` to an
#     [OT_FORM_Settings][gemseo.uncertainty.reliability.openturns.form_settings.OT_FORM_Settings]
#     to tune the FORM stage.
