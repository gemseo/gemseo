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
#        :author: François Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Multi-point computations using namespaces
=========================================
"""

from __future__ import annotations

from gemseo import configure_logger
from gemseo import create_design_space
from gemseo import create_discipline
from gemseo import create_scenario

configure_logger()
# %%
# About namespaces
# ----------------
# Namespaces are prefixes to input or output names of the disciplines.
# The name of a variable can be prefixed by the namespace and a separator, ":" by default.
# This allows to control the coupling of disciplines, or to duplicate a discipline
# in a process.

# %%
# Instantiate the disciplines
# ---------------------------
# We instantiate twice the same discipline, they will be configured with the namespaces
# to compute different outputs. Some inputs will be shared, without namespace, and others
# will have different namespaces.
#
disciplines = create_discipline(["SobieskiMission", "SobieskiMission"])

# %%
# Setting the namespaces
# ----------------------
# In the SSBJ test case, the ``y_34`` vector is the engine Specific Fuel Consumption (SFC).
# Let's say that we want to compute the average range of the aircraft, operating at
# two different operating conditions (``oc1`` and ``oc2``), which leads to two different SFC.
# We add the specific namespaces to the SFC (``y_34``) of the disciplines,
# and to the output range (``y_4``)
# The other inputs of the discipline remain the same, so they share their values.
disciplines[0].input_grammar.add_namespace("y_34", "oc1")
disciplines[1].input_grammar.add_namespace("y_34", "oc2")

disciplines[0].output_grammar.add_namespace("y_4", "oc1")
disciplines[1].output_grammar.add_namespace("y_4", "oc2")

# %%
# Averaging the outputs
# ---------------------
# Then we need to create a specific discipline to average the outputs.
# We use the LinearCombination discipline for that.
# Note that the namespaces can be set by the input name directly.

disciplines.append(
    create_discipline(
        "LinearCombination",
        input_names=["oc1:y_4", "oc2:y_4"],
        output_name="average_y_4",
        input_coefficients={"oc1:y_4": 0.5, "oc2:y_4": 0.5},
    )
)
# %%
# Build, execute and post-process the scenario
# --------------------------------------------
# In the design space, we add the two variables that are namespaced, and
# simulate several SFCs. The other variables remain at their default input's values.
design_space = create_design_space()
design_space.add_variable("oc1:y_34", lower_bound=0.8, upper_bound=2.0)
design_space.add_variable("oc2:y_34", lower_bound=0.8, upper_bound=2.0)

# %%
# Instantiate the scenario
# ^^^^^^^^^^^^^^^^^^^^^^^^
# The objective is the averaged range.
scenario = create_scenario(
    disciplines,
    formulation_name="MDF",
    objective_name="average_y_4",
    design_space=design_space,
    maximize_objective=True,
    scenario_type="DOE",
)
# %%
# Visualize the XDSM
# ^^^^^^^^^^^^^^^^^^
# The XDSM shows well the averaging proces and the duplication of the disciplines
# as well as the data handling
scenario.xdsmize(save_html=False)

scenario.execute(n_samples=20, algo_name="LHS")

# %%
# Plot the optimization history view
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
scenario.post_process(post_name="OptHistoryView", save=False, show=True)
