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
"""# Execute a scenario

## Problem

You want to execute an [MDOScenario][gemseo.scenarios.mdo.MDOScenario]
(or an [EvaluationScenario][gemseo.scenarios.evaluation.EvaluationScenario])
with a given algorithm.

## Solution

Each algorithm, along with its specific options, is defined through a dedicated model.
When a scenario is [executed][gemseo.scenarios.evaluation.EvaluationScenario.execute],
this model is provided
so that GEMSEO knows which algorithm to run and which options to apply.

In certain cases,
it may also be necessary to specify default settings for the algorithm.
Thus,
the
[set_algorithm()][gemseo.scenarios.evaluation.EvaluationScenario.set_algorithm] method
is useful.

## Step-by-step guide
"""

from __future__ import annotations

from gemseo.algos.design_space import DesignSpace
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.formulations.disciplinary_opt_settings import DisciplinaryOpt_Settings
from gemseo.scenarios.mdo import MDOScenario
from gemseo.settings.opt import L_BFGS_B_Settings
from gemseo.settings.opt import NLOPT_COBYLA_Settings
from gemseo.settings.opt import SLSQP_Settings

# %%
# ### 1. Create your scenario
#
# Here, an `AnalyticDiscipline` is used, with the `DisciplinaryOpt` formulation.
# An [MDOScenario][gemseo.scenarios.mdo.MDOScenario] is generated,
# but it would also work with an
# [EvaluationScenario][gemseo.scenarios.evaluation.EvaluationScenario].
discipline = AnalyticDiscipline(expressions={"y": "x1+x2"})
design_space = DesignSpace()
design_space.add_variable("x1", lower_bound=-5, upper_bound=5)
design_space.add_variable("x2", lower_bound=-5, upper_bound=5)

scenario = MDOScenario(
    (discipline,), design_space, formulation_settings=DisciplinaryOpt_Settings()
)

# %%
# and add an objective function
scenario.add_objective("y")

# %%
# !!! note
#     When creating an
#     [EvaluationScenario][gemseo.scenarios.evaluation.EvaluationScenario],
#     you don't have any objective function.
#
# ### 2. Execute with a given algorithm
#
# Select the settings
settings = SLSQP_Settings(max_iter=5)
settings

# %%
# ... and execute the scenario with these settings
scenario.execute(settings)

# %%
# ### 3. Set a default algorithm to the scenario
#
# In some particular cases, you may want to set a default algorithm.
#
# Here, define your settings.
default_settings = NLOPT_COBYLA_Settings(
    max_iter=10,
    ftol_rel=1e-10,
    ineq_tolerance=2e-3,
    normalize_design_space=True,
)
default_settings

# %%
# Set these settings as default in your scenario.
scenario.set_algorithm(default_settings)

# %%
# You can execute your scenario without giving any other settings.
scenario.execute()

# %%
# You can still use another algorithm, even if you set default before.
# By specifying an algorithm setting, you will override the default settings.
#
# Here, you don't use `NLOPT_COBYLA`.
scenario.execute(L_BFGS_B_Settings(max_iter=5))
# %%
# ## Summary
#
# You can choose the algorithm (and its associated options)
# by passing a dedicated settings model
# to the [execute()][gemseo.scenarios.mdo.MDOScenario.execute] method.
# Settings are provided directly at each
# [execute()][gemseo.scenarios.evaluation.EvaluationScenario.execute] call.
#
# In some cases,
# default settings model can be given beforehand to a specific scenario with the
# [set_algorithm()][gemseo.scenarios.evaluation.EvaluationScenario.set_algorithm]
# method.
