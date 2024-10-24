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
from gemseo.core.discipline import Discipline
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
    "MDF",
    "y_4",
    design_space,
    maximize_objective=True,
)

for constraint in ["g_1", "g_2", "g_3"]:
    scenario.add_constraint(constraint, constraint_type="ineq")

# %%
# Activate time stamps
# --------------------
# In order to record all time stamps recording, we have to call this method
# before the execution of the scenarios
ExecutionStatistics.is_time_stamps_enabled = True

scenario.execute(algo_name="SLSQP", max_iter=10)

# %%
# Post-process scenario
# ---------------------
# Lastly, we plot the Gantt chart.
create_gantt_chart(save=False, show=True)

# Finally, we deactivate the time stamps for other executions
Discipline.deactivate_time_stamps()
