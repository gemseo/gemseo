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
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Simple disciplinary DOE example on the Sobieski SSBJ test case
==============================================================
"""

from __future__ import annotations

from gemseo import configure_logger
from gemseo import create_discipline
from gemseo import create_scenario
from gemseo.problems.mdo.sobieski.core.design_space import SobieskiDesignSpace

configure_logger()


# %%
# Instantiate the discipline
# --------------------------
discipline = create_discipline("SobieskiMission")

# %%
# Create the design space
# -----------------------
design_space = SobieskiDesignSpace()
design_space.filter(["y_24", "y_34"])

# %%
# Create the scenario
# -----------------------
# Build scenario which links the disciplines with the formulation and
# The DOE algorithm.
scenario = create_scenario(
    [discipline],
    "y_4",
    design_space,
    maximize_objective=True,
    scenario_type="DOE",
    formulation_name="DisciplinaryOpt",
)

# %%
# Execute the scenario
# -----------------------
# Here we use a latin hypercube sampling algorithm with 30 samples.
scenario.execute(algo_name="PYDOE_LHS", n_samples=30)

# %%
# Note that both the formulation settings passed to func:`.create_scenario` and the
# algorithm settings passed to :meth:`~.DriverLibrary.execute` can be provided via a Pydantic model. For
# more information, see :ref:`formulation_settings` and :ref:`algorithm_settings`.
#
# Plot optimization history view
# ------------------------------
scenario.post_process(post_name="OptHistoryView", save=False, show=True)

# %%
# Note that post-processor settings passed to :meth:`.BaseScenario.post_process` can be
# provided via a Pydantic model (see the example below). For more information,
# see :ref:`post_processor_settings`.
#
# Plot scatter plot matrix
# ------------------------
from gemseo.settings.post import ScatterPlotMatrix_Settings  # noqa: E402

settings_model = ScatterPlotMatrix_Settings(
    variable_names=["y_4", "y_24", "y_34"],
    save=False,
    show=True,
)

scenario.post_process(settings_model)
