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
"""# Run a reliability analysis

## Problem

You want to estimate the probability that a discipline output crosses a threshold,
given uncertain inputs described by probability distributions.

## Solution

[ReliabilityScenario][gemseo.uncertainty.reliability.scenario.ReliabilityScenario]
wraps a list of disciplines and an uncertain space
into a [ReliabilityProblem][gemseo.uncertainty.reliability.problem.ReliabilityProblem].
The workflow is:

1. Create a
   [ReliabilityScenario][gemseo.uncertainty.reliability.scenario.ReliabilityScenario]
   from a list of disciplines and an uncertain space.
2. Define one or more events using comparison operators on disciplinary outputs.
3. Execute the scenario with a reliability algorithm settings object,
   e.g. [OT_FORM_Settings][gemseo.uncertainty.reliability.openturns.form_settings.OT_FORM_Settings].
4. Read the estimated probability from the returned
   [ReliabilityResult][gemseo.uncertainty.reliability.result.ReliabilityResult].

## Step-by-step guide
"""

from __future__ import annotations

from gemseo.problem.uncertainty.wing_weight.discipline import WingWeightDiscipline
from gemseo.problem.uncertainty.wing_weight.uncertain_space import (
    WingWeightUncertainSpace,
)
from gemseo.uncertainty.reliability.openturns.form_settings import OT_FORM_Settings
from gemseo.uncertainty.reliability.scenario import ReliabilityScenario

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
# ### 2. Create a reliability scenario
#
# [ReliabilityScenario][gemseo.uncertainty.reliability.scenario.ReliabilityScenario]
# takes the list of disciplines and the uncertain space:
scenario = ReliabilityScenario([discipline], uncertain_space)

# %%
# ### 3. Define an event
#
# First, create an event variable bound to the disciplinary output `Ww`:
Ww = scenario.get_event_variables("Ww")

# %%
# !!! tip
#     You can combine events using logical and comparison operators,
#     e.g. `(a > 3) | (b < 12) & (c.isin([-6, 9]))`
#     Note: parentheses are required for elementary events
#     and `&` (AND) takes precedence over `|` (OR).
#
# Then define the event of interest named `"too_heavy"` using a comparison operator.
# Here the event is that the wing weight exceeds 400 lb:
scenario.add_event(Ww > 400, "too_heavy")

# %%
# ### 4. Execute with FORM
#
# [OT_FORM_Settings][gemseo.uncertainty.reliability.openturns.form_settings.OT_FORM_Settings]
# selects the first-order reliability method (FORM) from OpenTURNS.
# [execute()][gemseo.scenario.evaluation.EvaluationScenario.execute]
# returns a mapping from event names to
# [ReliabilityResult][gemseo.uncertainty.reliability.result.ReliabilityResult] objects:
results = scenario.execute(OT_FORM_Settings())

# %%
# ### 5. Read the results
#
# Get the result for the `"too_heavy"` event:
event_result = results["too_heavy"]

# %%
# The estimated probability of the event:
event_result.probability

# %%
# The raw OpenTURNS FORM result, for advanced post-processing:
event_result.raw_result

# %%
# ## Summary
#
# - [ReliabilityScenario][gemseo.uncertainty.reliability.scenario.ReliabilityScenario]
#   wraps disciplines and an uncertain space;
# - [get_event_variables()][gemseo.uncertainty.reliability.scenario.ReliabilityScenario.get_event_variables]
#   creates symbolic variables bound to disciplinary outputs;
# - [add_event()][gemseo.uncertainty.reliability.scenario.ReliabilityScenario.add_event]
#   registers an event built from comparison operators (`>`, `<`, `>=`, `<=`);
# - [execute()][gemseo.scenario.evaluation.EvaluationScenario.execute]
#   runs the algorithm and returns a `dict[str, ReliabilityResult]`;
# - [ReliabilityResult.probability][gemseo.uncertainty.reliability.result.ReliabilityResult.probability]
#   holds the estimated failure probability;
# - [ReliabilityResult.raw_result][gemseo.uncertainty.reliability.result.ReliabilityResult.raw_result]
#   exposes the underlying library object for advanced post-processing.
