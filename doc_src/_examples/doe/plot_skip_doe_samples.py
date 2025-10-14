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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Gilberto Ruiz Jiménez
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Skip samples when using DOE
===========================
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import array
from numpy import sqrt

from gemseo import create_design_space
from gemseo import create_scenario
from gemseo.core.discipline import Discipline
from gemseo.settings.doe import CustomDOE_Settings
from gemseo.settings.post import BasicHistory_Settings

if TYPE_CHECKING:
    from gemseo.typing import StrKeyMapping


# %%
# In this example, we show how to skip the evaluation at a DOE point during a
# :class:`.DOEScenario` run. This is useful in situations where the evaluation of a
# sample fails or when the user wants to avoid evaluating some samples when
# certain conditions are met.
# The DOE algorithms in GEMSEO are able to catch ``ValueError`` exceptions
# (and only this specific type of exception) at runtime and
# move to the next sample.

# %%
# Let us consider a discipline implementing the function :math:`y=sqrt(a)`. The ``_run``
# method of this discipline raises a ``ValueError`` when :math:`a < 0`. Of course,
# you may use any other set of conditions to raise the exception in your scripts.


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
# We define a design space with the variable :math:`a\in[-1,10]`:
design_space = create_design_space()
design_space.add_variable(
    "a", type_=design_space.DesignVariableType.FLOAT, lower_bound=-1, upper_bound=10
)

# %%
# We want to evaluate this discipline over this design space
# at points 1, -1 and 4:
samples = array([[1.0], [-1.0], [4.0]])

# %%
# For that, we can create a scenario and execute it with a :class:`.CustomDOE`
# with the setting "samples":
scenario = create_scenario(
    [discipline],
    "y",
    design_space,
    scenario_type="DOE",
    formulation_name="DisciplinaryOpt",
)
custom_doe_settings = CustomDOE_Settings(samples=samples)
scenario.execute(custom_doe_settings)

# %%
# The logger shows that
# GEMSEO ignores the ``ValueError`` raised at the second point
# and switches to the third point.
# The post-processing of the scenario only includes two runs:
basic_history_settings = BasicHistory_Settings(
    variable_names=["y"], save=False, show=True
)
scenario.post_process(basic_history_settings)

# %%
# .. warning::
#     In order to be able to continue the execution of a :class:`.DOEScenario`, the
#     execution status and statistics of disciplines must be disabled. This is the
#     default behavior, but you can also disable them explicitly with the
#     :func:`.configure` function.

# %%
# Note that if your discipline is able to return a
# ``NaN`` without raising any exceptions, you do not need to use the mechanism
# explained here, in that case the ``NaN`` values will be handled by GEMSEO and stored
# in the database of the scenario.
