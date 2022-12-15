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

from gemseo.api import configure_logger
from gemseo.api import create_design_space
from gemseo.api import create_discipline
from gemseo.api import create_scenario
from gemseo.api import get_available_scenario_types
from gemseo.api import get_scenario_differentiation_modes
from gemseo.api import get_scenario_inputs_schema
from gemseo.api import get_scenario_options_schema
from gemseo.api import monitor_scenario

configure_logger()


##########################################################################
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
# The API function :meth:`~gemseo.api.get_available_scenario_types` can be used
# to get the available scenario types (:class:`.MDOScenario` and
# :class:`.DOEScenario`).
print(get_available_scenario_types())

##########################################################################
# Get scenario options schema
# ---------------------------
# The :meth:`~gemseo.api.get_scenario_options_schema` function can be used
# to get the options of a given scenario type:
print(get_scenario_options_schema("MDO"))

##########################################################################
# Create a scenario
# -----------------
# The API function :meth:`~gemseo.api.create_scenario` can be used
# to create a scenario:
#
# - The four first arguments are mandatory:
#
#   - :code:`disciplines`: the list of :class:`.MDODiscipline`
#     (or possibly, a single :class:`.MDODiscipline`),
#   - :code:`formulation`: the formulation name,
#   - :code:`objective_name`: the name of the objective function
#     (one of the discipline outputs)
#   - :code:`design_space`: the :class:`.DesignSpace` or
#     the file path of the design space
#
# - The other arguments are optional:
#
#   - :code:`name`: scenario name,
#   - :code:`scenario_type`: type of scenario,
#     either `"MDO"` (default) or `"DOE"` ,
#   - :code:`**options`: options passed to the formulation.
#
# - This function returns an instance of :class:`.MDOScenario` or
#   :class:`.DOEScenario`.

discipline = create_discipline("AnalyticDiscipline", expressions={"y": "x1+x2"})
design_space = create_design_space()
design_space.add_variable("x1", 1, "float", 0.0, 1.0)
design_space.add_variable("x2", 1, "float", 0.0, 1.0)

scenario = create_scenario(
    discipline, "DisciplinaryOpt", "y", design_space, scenario_type="DOE"
)
scenario.execute({"algo": "fullfact", "n_samples": 25})
scenario.post_process(
    "ScatterPlotMatrix",
    variable_names=["x1", "x2", "y"],
    save=False,
    show=True,
)

##########################################################################
# - The :meth:`~gemseo.api.get_scenario_inputs_schema` function can be used
#   to get the inputs of a scenario:
print(get_scenario_inputs_schema(scenario))

##########################################################################
# Get scenario differentiation modes
# ----------------------------------
# The :meth:`~gemseo.api.get_scenario_differentiation_modes` can be used to
# get the available differentiation modes of a scenario:
print(get_scenario_differentiation_modes())

##########################################################################
# Monitor a scenario
# ------------------
# To monitor a scenario execution programmatically,
# ie get a notification when a discipline status is changed,
# use :meth:`~gemseo.api.monitor_scenario`.
# The first argument is the scenario to monitor, and the second is an
# observer object, that is notified by its update(atom) method, which takes an
# :class:`~gemseo.core.execution_sequence.AtomicExecSequence` as argument.
# This method will be called every time a discipline status changes.
# The atom represents a discipline's position in the process. One discipline
# can have multiple atoms, since one discipline can be used in multiple
# positions in the MDO formulation.


class Observer:
    """Observer."""

    def update(self, atom):
        """Update method.

        :param AtomicExecSequence atom: atomic execution sequence.
        """
        print(atom)


scenario = create_scenario(
    discipline, "DisciplinaryOpt", "y", design_space, scenario_type="DOE"
)
monitor_scenario(scenario, Observer())
scenario.execute({"algo": "fullfact", "n_samples": 25})
