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
"""# Tutorial - Execute your first Design of Experiment (DoE)

## Goal

In this tutorial,
you will create an [EvaluationScenario][gemseo.scenarios.evaluation.EvaluationScenario]
to evaluate a discipline in multiple points.
"""

from __future__ import annotations

from gemseo.algos.design_space import DesignSpace
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.scenarios.evaluation import EvaluationScenario
from gemseo.settings.doe import PYDOE_FULLFACT_Settings

# %%
# ## Step 1 — The discipline to evaluate
#
# Firstly,
# we create a [Discipline][gemseo.core.discipline.discipline.Discipline] of
# [AnalyticDiscipline][gemseo.disciplines.analytic.AnalyticDiscipline] type
# from a Python function:
discipline = AnalyticDiscipline({"y": "x1+x2"})

# %%
# ## Step 2 - Define your design space
#
# Now, we want to evaluate this
# [Discipline][gemseo.core.discipline.discipline.Discipline]
# over a design of experiments (DOE).
# The points to evaluate will be chosen in a given
# [DesignSpace][gemseo.algos.design_space.DesignSpace] that you will define.
#
# Here, you want to constraint your integer design variables
# $(x1, x2) \in [-5, 5]\times[-5, 5]$
# by using its
# [add_variable()][gemseo.algos.design_space.DesignSpace.add_variable] method.

design_space = DesignSpace()
design_space.add_variable("x1", lower_bound=-5, upper_bound=5, type_="integer")
design_space.add_variable("x2", lower_bound=-5, upper_bound=5, type_="integer")

# %%
# ## Step 3 - Define your DoE scenario
#
# Then,
# we define an [EvaluationScenario][gemseo.scenarios.evaluation.EvaluationScenario]
# from the [Discipline][gemseo.core.discipline.discipline.Discipline]
# and the [DesignSpace][gemseo.algos.design_space.DesignSpace] defined above:

scenario = EvaluationScenario((discipline,), design_space, name="My evaluation")

# %%
# !!! note
#     Here,
#     we chose the `DisciplinaryOpt` formulation since we get only one discipline.
#     Other formulations can be chosen for more complex evaluation workflows.
#
# We specify which variables to observe.
# In our case,
# the `y` variable is important.
# We chose to give this variable a new name (`Result`),
# for plot purposes.
scenario.add_observable("y", observable_name="Result")
# %%
# ## Step 4 - Execute your DoE scenario
#
# The sampling strategy of our
# [DesignSpace][gemseo.algos.design_space.DesignSpace],
# is determinded through the selection of a DoE algorithm.
# In our case,
# since all possible combinations correspond to only 121 points,
# we want to evaluate all of them.
# To this end,
# we choose a
# [full factorial design](https://en.wikipedia.org/wiki/Factorial_experiment)
# of size $11^2$:

scenario.execute(PYDOE_FULLFACT_Settings(n_samples=11**2))
# %%
# !!! note
#     You can also give explicitely the points to evaluate with the
#     [CustomDOE_Settings][gemseo.algos.doe.custom_doe.settings.custom_doe_settings.CustomDOE_Settings]
#     pydantic model.
#
# ## Step 5 - Retrieve the results
#
# When the scenario has evaluated all the points,
# you can retrieve the results with the generation of a
# [Dataset][gemseo.datasets.dataset.Dataset].
#
# Two different groups are made in the [Dataset][gemseo.datasets.dataset.Dataset]:
#
# - `inputs` to store the evaluated points
# - `outputs` to store the observables.
#
# In our case, the `y` variable appears as `Result`.
scenario.to_dataset()

# %%
# ## Key takeaways
#
# In this tutorial you've learnt to agregate the notions of
# [Discipline][gemseo.core.discipline.discipline.Discipline],
# [DesignSpace][gemseo.algos.design_space.DesignSpace] and
# [EvaluationScenario][gemseo.scenarios.evaluation.EvaluationScenario]
# to create a simple process which evaluates your workflow in multiple points.
#
# You will learn later
# that a Design of Experiment can also be used to solve an optimization problem.
# For that, please refer to the next tutorial:
# [Tutorial - Execute your first Multi-Disciplinary Optimization][tutorial-execute-your-first-multi-disciplinary-optimization]
#
# ## How-to guides
#
# For further information,
# please refer to the following how-to guides:
#
# - [Use multi-processing for your DOE][use-multi-processing-for-your-doe]
# - [Save the execution history of a scenario][save-the-execution-history-of-a-scenario]
