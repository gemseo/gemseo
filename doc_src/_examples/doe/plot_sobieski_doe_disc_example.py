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

from gemseo.api import configure_logger
from gemseo.api import create_discipline
from gemseo.api import create_scenario
from gemseo.problems.sobieski.core.problem import SobieskiProblem

configure_logger()


##############################################################################
# Instantiate the discipline
# --------------------------
discipline = create_discipline("SobieskiMission")

##############################################################################
# Create the design space
# -----------------------
design_space = SobieskiProblem().design_space
design_space.filter(["y_24", "y_34"])

##############################################################################
# Create the scenario
# -----------------------
# Build scenario which links the disciplines with the formulation and
# The DOE algorithm.
scenario = create_scenario(
    [discipline],
    formulation="DisciplinaryOpt",
    objective_name="y_4",
    design_space=design_space,
    maximize_objective=True,
    scenario_type="DOE",
)

##############################################################################
# Execute the scenario
# -----------------------
# Here we use a latin hypercube sampling algorithm with 30 samples.
scenario.execute({"n_samples": 30, "algo": "lhs"})

##############################################################################
# Plot optimization history view
# ------------------------------
scenario.post_process("OptHistoryView", save=False, show=True)

##############################################################################
# Plot parallel coordinates
# -------------------------
scenario.post_process(
    "ScatterPlotMatrix",
    variable_names=["y_4", "y_24", "y_34"],
    save=False,
    show=True,
)
