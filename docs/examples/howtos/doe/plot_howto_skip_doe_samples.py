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
"""# Skip samples during a DOE

## Problem

During a DOE execution, some samples may be invalid
(e.g. a function is undefined for certain input values).
By default, an unhandled exception would stop the entire run.

## Solution

Raise a `ValueError` inside the discipline's `_run` method
when a sample should be skipped.
GEMSEO catches `ValueError` exceptions at runtime and moves on to the next sample,
leaving the invalid point out of the results.

## Step-by-step guide
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import array
from numpy import sqrt

from gemseo.algos.design_space import DesignSpace
from gemseo.core.discipline import Discipline
from gemseo.scenarios.evaluation import EvaluationScenario
from gemseo.settings import CustomDOE_Settings

if TYPE_CHECKING:
    from gemseo.typing import StrKeyMapping

# %%
# ### 1. Implement the discipline
#
# Raise a `ValueError` in `_run` whenever a sample must be skipped.
# Here the function $y = \sqrt{a}$ is undefined for $a < 0$:


class ValueErrorDiscipline(Discipline):
    default_grammar_type = Discipline.GrammarType.SIMPLE

    def __init__(self) -> None:
        super().__init__()
        self.input_grammar.update_from_names("a")
        self.output_grammar.update_from_names("y")

    def _run(self, input_data: StrKeyMapping):
        a = input_data["a"]
        if a < 0:
            msg = "The sample is undefined for a < 0."
            raise ValueError(msg)
        return {"y": sqrt(a)}


discipline = ValueErrorDiscipline()

# %%
# ### 2. Create an EvaluationScenario
#
design_space = DesignSpace()
design_space.add_variable("a", lower_bound=-1.0, upper_bound=10.0)

# %%
# For that, we can create a scenario and execute it with a [CustomDOE][gemseo.algos.doe.custom_doe.custom_doe.CustomDOE]
# with the setting "samples":
scenario = EvaluationScenario([discipline], design_space)
scenario.add_observable("y")
# %%
# ### 3. Evaluate the scenario
#
# We want to evaluate this discipline over this design space
# at points 1, -1 and 4:
samples = array([[1.0], [-1.0], [4.0]])
scenario.execute(CustomDOE_Settings(samples=samples))

# %%
# The logger shows that
# GEMSEO ignores the `ValueError` raised at the second point
# and switches to the third point.
# The database will only contain 2 points.
scenario.to_dataset()
# %%
# !!! warning
#     In order to be able to continue the execution of a scenario,
#     the execution status and statistics of disciplines must be disabled.
#     This is the default behavior,
#     but you can also disable them explicitly with the
#     [configure()][gemseo.configure] function.
#
# !!! note
#     If your discipline is able to return a
#     `NaN` without raising any exceptions, you do not need to use the mechanism
#     explained here, in that case the `NaN` values will be handled by GEMSEO and stored
#     in the database of the scenario.
#
# ## Summary
#
# When using a design of experiments,
# GEMSEO catches any `ValueError` exceptions raised by disciplines at runtime
# and moves on to the next sample,
# leaving the invalid point out of the results.
# The database will not contain invalid points,
# so that every post-process will only contain valid points.
