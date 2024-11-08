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
"""
Scenario
========
"""

from __future__ import annotations

from gemseo import configure_logger
from gemseo import create_design_space
from gemseo import create_discipline
from gemseo import create_scenario
from gemseo import get_available_scenario_types
from gemseo import get_scenario_differentiation_modes
from gemseo import get_scenario_inputs_schema
from gemseo import get_scenario_options_schema
from gemseo import monitor_scenario

configure_logger()


# %%
# In this example, we will discover the different functions of the API to
# related to scenarios, which are the |g|' objects
# dedicated to the resolution of a problem, e.g. optimization or trade-off,
# associated with a list of disciplines and a design space. All classes
# implementing scenarios inherit from :class:`.Scenario` which is an
# abstract class. Classical concrete classes are :class:`.MDOScenario` and
# :class:`.DOEScenario`, respectively dedicated to optimization and trade-off
# problems.
#
# Get available scenario type
# ---------------------------
# The high-level function :func:`~gemseo.get_available_scenario_types` can be used
# to get the available scenario types (:class:`.MDOScenario` and
# :class:`.DOEScenario`).
get_available_scenario_types()

# %%
# Get scenario options schema
# ---------------------------
# The :func:`.get_scenario_options_schema` function can be used
# to get the options of a given scenario type:
get_scenario_options_schema("MDO")

# %%
# Create a scenario
# -----------------
# The API function :func:`.create_scenario` can be used
# to create a scenario:
#
# - The four first arguments are mandatory:
#
#   - ``disciplines``: the list of :class:`.Discipline`
#     (or possibly, a single :class:`.Discipline`),
#   - ``objective_name``: the name of the objective function
#     (one of the discipline outputs),
#   - ``design_space``: the :class:`.DesignSpace` or
#     the file path of the design space,
#   - either a ``formulation_name`` followed by its ``formulation_settings```; or
#   - a ``formulation_settings_model`` (see :ref:`formulation_settings`).
#
# - The other arguments are optional:
#
#   - ``name``: scenario name,
#   - ``scenario_type``: type of scenario,
#     either `"MDO"` (default) or `"DOE"` ,
#   - ``**formulation_settings``: settings passed to the formulation as keyword
#     arguments when the ``formulation_settings_model`` was not provided.
#
# - This function returns an instance of :class:`.MDOScenario` or
#   :class:`.DOEScenario`.

discipline = create_discipline("AnalyticDiscipline", expressions={"y": "x1+x2"})
design_space = create_design_space()
design_space.add_variable("x1", 1, "float", 0.0, 1.0)
design_space.add_variable("x2", 1, "float", 0.0, 1.0)

scenario = create_scenario(
    discipline,
    "y",
    design_space,
    scenario_type="DOE",
    formulation_name="DisciplinaryOpt",
)
scenario.execute(algo_name="PYDOE_FULLFACT", n_samples=25)
scenario.post_process(
    post_name="ScatterPlotMatrix",
    variable_names=["x1", "x2", "y"],
    save=False,
    show=True,
)

# %%
# - The :func:`.get_scenario_inputs_schema` function can be used
#   to get the inputs of a scenario:
get_scenario_inputs_schema(scenario)

# %%
# Get scenario differentiation modes
# ----------------------------------
# The :func:`.get_scenario_differentiation_modes` can be used to
# get the available differentiation modes of a scenario:
get_scenario_differentiation_modes()

# %%
# Monitor a scenario
# ------------------
# To monitor a scenario execution programmatically,
# ie get a notification when a discipline status is changed,
# use :func:`.monitor_scenario`.
# The first argument is the scenario to monitor, and the second is an
# observer object, that is notified by its update(atom) method, which takes an
# :class:`~gemseo.core.execution_sequence.ExecutionSequence` as argument.
# This method will be called every time a discipline status changes.
# The atom represents a discipline's position in the process. One discipline
# can have multiple atoms, since one discipline can be used in multiple
# positions in the MDO formulation.


class Observer:
    """Observer."""

    def update(self, atom) -> None:
        """Update method.

        :param ExecutionSequence atom: atomic execution sequence.
        """
        print(atom)


scenario = create_scenario(
    discipline,
    "y",
    design_space,
    scenario_type="DOE",
    formulation_name="DisciplinaryOpt",
)
monitor_scenario(scenario, Observer())
scenario.execute(algo_name="PYDOE_FULLFACT", n_samples=25)
