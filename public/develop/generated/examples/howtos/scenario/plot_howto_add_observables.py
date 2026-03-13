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
"""# Observe variables of interest

## Problem

You have executed an MDO scenario and you can retrieve the values of your design variables,
constraints and objective functions.
However, you are also interested in some other discipline outputs.

How can you tell your scenario to store these output values at each iteration?

## Solution

When creating your MDO scenario,
you can add observables using the
[add_observable()][gemseo.scenarios.mdo.MDOScenario.add_observable] method.

## Step-by-step guide
"""

from __future__ import annotations

from numpy import array
from numpy import ones

from gemseo import create_discipline
from gemseo import create_scenario
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt.scipy_local.settings.slsqp import SLSQP_Settings
from gemseo.formulations.mdf_settings import MDF_Settings

# %%
# ### 1. Define your scenario
#
# Let's consider the Sellar's problem, defined with two constraints.
disciplines = create_discipline(["Sellar1", "Sellar2", "SellarSystem"])

design_space = DesignSpace()
design_space.add_variable("x_local", lower_bound=0.0, upper_bound=10.0, value=ones(1))
design_space.add_variable(
    "x_shared",
    2,
    lower_bound=(-10, 0.0),
    upper_bound=(10.0, 10.0),
    value=array([4.0, 3.0]),
)
scenario = create_scenario(
    disciplines, "obj", design_space, formulation_settings_model=MDF_Settings()
)
scenario.add_constraint("c_1", constraint_type="ineq")
scenario.add_constraint("c_2", constraint_type="ineq")

# %%
# ### 2. Add observables
#
# Only the design variables, objective function and constraints are stored by
# default in the history database.
# In order to also store any extra variable, you can add it as an observable.
# All we have to do is enter
# the variable name as a string to the
# [add_observable()][gemseo.scenarios.mdo.MDOScenario.add_observable].
# If more than one output name is provided (as a list of strings),
# the observable function returns a concatenated array of the output values.
scenario.add_observable("y_1")
# %%
# It is also possible to add the observable with a custom name,
# using the option `observable_name`. Let us store the variable `y_2` as `y2`.
scenario.add_observable("y_2", observable_name="y2")

# %%
# ### 3. Execute the scenario
#
# Then,
# we execute the MDO scenario with the inputs of the MDO scenario as a dictionary.
# In this example,
# the gradient-based `SLSQP` optimizer is selected, with 10 iterations at maximum:
scenario.execute(SLSQP_Settings(max_iter=10))
scenario.to_dataset()

# %%
# ## Summary
#
# Observables can be added to an MDO scenario with the
# [add_observable()][gemseo.scenarios.mdo.MDOScenario.add_observable] method.
# The execution of your scenario will then store
# the given observed variables into the scenario database.
