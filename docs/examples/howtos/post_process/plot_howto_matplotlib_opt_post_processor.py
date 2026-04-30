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
"""# Customise a post-processing figure with matplotlib

## Problem

The settings of a post-processor cover the most common formatting needs,
but some fine-tuning (e.g. axis labels, colour bar labels) requires
direct access to the underlying matplotlib figures.

## Solution

Retrieve the matplotlib figures from the post-processor via its `figures`
attribute, then use the standard matplotlib API to modify them before saving.

## Step-by-step guide
"""

from __future__ import annotations

from matplotlib import pyplot as plt

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt.nlopt.settings.nlopt_cobyla_settings import NLOPT_COBYLA_Settings
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.formulations.disciplinary_opt_settings import DisciplinaryOpt_Settings
from gemseo.post import OptHistoryView_Settings
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
# ### 2. Run the post-processing and retrieve the figures
#
# Pass `save=False` and `show=False` to prevent GEMSEO from saving or
# displaying the figures immediately, so they can be modified first:
opt_post_processor = scenario.post_process(
    OptHistoryView_Settings(save=False, show=False)
)
figures = opt_post_processor.figures
figures

# %%
# !!! tip
#     Figures are stored in the `figure` dictionary attribute.
#     The keys vary depending on the post-processor used;
#     print them to discover how to access each figure.
#
# ### 3. Customise the figure with matplotlib
#
# Access the axes of the `"variables"` figure and update the labels:
figure = figures["variables"]
axes = figure.axes
axes[1].set_ylabel("Optimization variables scaled in [0,1]")
axes[0].set_ylabel("Optimization variables")

# %%
# ### 4. Display the modified figure
#
plt.figure(figure)
plt.show()


# %%
# !!! tip
#     You can save the figure with `figure.savefig("variables.png")`.`
#
# ## Summary
#
# Retrieve matplotlib figures from the post-processor via `post_processor.figures`,
# then modify them through `figure.axes` before calling `plt.show()` or
# `figure.savefig()`.
# Always check first whether a post-processor setting already covers the needed
# customisation; fall back to the matplotlib API only when it does not.
