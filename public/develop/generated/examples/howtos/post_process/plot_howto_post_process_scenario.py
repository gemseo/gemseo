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
"""# Post-process a scenario

## Problem

After executing a scenario, you need to visualise the optimisation history.

## Solution

Use either the scenario method
[post_process()][gemseo.scenarios.mdo.MDOScenario.post_process]
or the function [execute_post()][gemseo.execute_post],
both accepting a post-processing settings model.

## Step-by-step guide
"""

from __future__ import annotations

from gemseo import execute_post
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt.nlopt.settings.nlopt_cobyla_settings import NLOPT_COBYLA_Settings
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.formulations.disciplinary_opt_settings import DisciplinaryOpt_Settings
from gemseo.post import BasicHistory_Settings
from gemseo.scenarios.mdo import MDOScenario

# %%
# ### 1. Build and execute the scenario
#
discipline = AnalyticDiscipline(expressions={"y": "x**2"})

design_space = DesignSpace()
design_space.add_variable("x", lower_bound=0.0, upper_bound=1.0)

scenario = MDOScenario(
    [discipline], design_space, formulation_settings=DisciplinaryOpt_Settings()
)
scenario.add_objective("y")
scenario.execute(NLOPT_COBYLA_Settings(max_iter=10))

# %%
# ### 2. Post-process the scenario
#
# Use the scenario method [post_process()][gemseo.scenarios.mdo.MDOScenario.post_process]:
scenario.post_process(BasicHistory_Settings(variable_names=["y"], save=False))

# %%
# or equivalently, the function [execute_post()][gemseo.execute_post]:
execute_post(
    scenario, BasicHistory_Settings(variable_names=["y"], save=False, show=True)
)

# %%
# !!! note
#     By default, GEMSEO saves the images on the disk.
#     Use `save=False` to not save figures and `show=True` to display them on the screen.

# %%
# ## Summary
#
# After executing a scenario, visualise the history by calling
# [post_process()][gemseo.scenarios.mdo.MDOScenario.post_process] on the scenario
# or [execute_post()][gemseo.execute_post] with the scenario as the first argument.
# Both methods accept the same post-processing settings models.
