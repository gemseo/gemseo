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
Gantt Chart
===========

In this example, we illustrate the use of the Gantt chart plot
on the Sobieski's SSBJ problem.
"""

# %%
# Import
# ------
# The first step is to import some high-level functions
# and a method to get the design space.
from __future__ import annotations

from gemseo import configure_logger
from gemseo import create_discipline
from gemseo import create_scenario
from gemseo.core.execution_statistics import ExecutionStatistics
from gemseo.post.core.gantt_chart import create_gantt_chart
from gemseo.problems.mdo.sobieski.core.design_space import SobieskiDesignSpace

configure_logger()


# %%
# Create disciplines
# ------------------
# Then, we instantiate the disciplines of the Sobieski's SSBJ problem:
# Propulsion, Aerodynamics, Structure and Mission
disciplines = create_discipline([
    "SobieskiPropulsion",
    "SobieskiAerodynamics",
    "SobieskiStructure",
    "SobieskiMission",
])

# %%
# Create design space
# -------------------
# We also create the :class:`.SobieskiDesignSpace`.
design_space = SobieskiDesignSpace()

# %%
# Create and execute scenario
# ---------------------------
# The next step is to build an MDO scenario in order to maximize the range,
# encoded 'y_4', with respect to the design parameters, while satisfying the
# inequality constraints 'g_1', 'g_2' and 'g_3'. We can use the MDF formulation,
# the SLSQP optimization algorithm
# and a maximum number of iterations equal to 100.
scenario = create_scenario(
    disciplines,
    "y_4",
    design_space,
    formulation_name="MDF",
    maximize_objective=True,
)

# %%
# Note that the formulation settings passed to :func:`.create_scenario` can be provided
# via a Pydantic model. For more information, see :ref:`formulation_settings`.

for constraint in ["g_1", "g_2", "g_3"]:
    scenario.add_constraint(constraint, constraint_type="ineq")

# %%
# Enable time stamps
# ------------------
# Recording all time stamps is done by default;
# we have to enable it:
ExecutionStatistics.is_time_stamps_enabled = True

scenario.execute(algo_name="SLSQP", max_iter=10)

# %%
# Note that the algorithm settings passed to :meth:`.DriverLibrary.execute` can be provided
# via a Pydantic model. For more information, see :ref:`algorithm_settings`.

# %%
# Post-process scenario
# ---------------------
# Lastly, we plot the Gantt chart.
create_gantt_chart(save=False, show=True)

# Finally, we disable the recording of time stamps for other executions:
ExecutionStatistics.is_time_stamps_enabled = False
