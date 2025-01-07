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
Execution statistics as a Gantt chart
=====================================

When
the global attribute :attr:`.ExecutionStatistics.is_time_stamps_enabled` is ``True``
(default: ``False``),
the global attribute :attr:`.ExecutionStatistics.time_stamps` is
a dictionary of the form ``{name: (initial_time, final_time, is_linearization)}``
to store
the initial and final times of each execution and linearization of each discipline.

The :func:`.create_gantt_chart` function can display this dictionary
in the form of a `Gantt chart <https://en.wikipedia.org/wiki/Gantt_chart>`__.

In this example,
we illustrate the use of this function
on the Sobieski's SSBJ problem.
"""

from __future__ import annotations

from gemseo import configure_logger
from gemseo import create_discipline
from gemseo import create_scenario
from gemseo.core.execution_statistics import ExecutionStatistics
from gemseo.post.core.gantt_chart import create_gantt_chart
from gemseo.problems.mdo.sobieski.core.design_space import SobieskiDesignSpace

configure_logger()


# %%
# Create the scenario
# ------------------
# First, we define the Sobieski's SSBJ problem as a scenario.
#
# For this,
# we instantiate the disciplines:
disciplines = create_discipline([
    "SobieskiPropulsion",
    "SobieskiAerodynamics",
    "SobieskiStructure",
    "SobieskiMission",
])

# %%
# as well as the design space:
design_space = SobieskiDesignSpace()

# %%
# Then,
# given these disciplines and design space,
# we build an MDO scenario using the MDF formulation
# in order to maximize the range ``"y_4"`` with respect to the design variables:
scenario = create_scenario(
    disciplines,
    "y_4",
    design_space,
    formulation_name="MDF",
    maximize_objective=True,
)

# %%
# and satisfy the inequality constraints
# associated with the outputs ``"g_1"``, ``"g_2"`` and ``"g_3"``:
for constraint in ["g_1", "g_2", "g_3"]:
    scenario.add_constraint(constraint, constraint_type="ineq")

# %%
# Execute the scenario
# --------------------
# By default,
# a scenario does *not* produce execution statistics.
# We need to enable this *global* mechanism before executing the scenario:
ExecutionStatistics.is_time_stamps_enabled = True
# %%
# .. warning::
#    This mechanism is *global*
#    and shall be modified from the :class:`.ExecutionStatistics` class
#    (not from an :class:`.ExecutionStatistics` instance).
#
# The scenario can now be executed
# using the SLSQP optimization algorithm and a maximum of 10 iterations:
scenario.execute(algo_name="SLSQP", max_iter=10)

# %%
# .. seealso::
#    The formulation settings passed to :func:`.create_scenario`
#    and the algorithm settings passed to :meth:`.BaseScenario.execute`
#    can be provided via Pydantic models.
#    For more information,
#    see :ref:`algorithm_settings` and :ref:`formulation_settings`.
#
# Plot the Gantt chart
# --------------------
# Lastly,
# we plot the Gantt chart from the global :attr:`.ExecutionStatistics.time_stamps`:
create_gantt_chart(save=False, show=True)
# %%
# This graph shows the evolution over time:
#
# - the execution and linearizations of the different user disciplines,
#   *e.g.* :class:`~gemseo.problems.mdo.sobieski.disciplines.SobieskiAerodynamics`,
# - the execution and linearizations of the different process disciplines,
#   *e.g.* :class:`.MDAJacobi`,
# - the execution of the scenario.

# %%
# Disable recording
# -----------------
# Finally,
# we disable the recording of time stamps for other executions:
ExecutionStatistics.is_time_stamps_enabled = False
# %%
# .. note::
#    As this reset :attr:`.ExecutionStatistics.time_stamps` to ``None``,
#    the :func:`.create_gantt_chart` function can no longer be used.
#    Set :attr:`.ExecutionStatistics.is_time_stamps_enabled` to ``True``
#    and execute or linearize some disciplines
#    so that you can use it again.
