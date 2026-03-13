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
"""# Use multi-processing for your DOE

## Problem

You want to execute a DOE algorithm in parallel.

## Solution

Multi-processing features are available when executing a DOE algorithm.

## Step-by-step guide
"""

from __future__ import annotations

from gemseo import create_discipline
from gemseo.formulations.mdf_settings import MDF_Settings
from gemseo.problems.mdo.sobieski.core.design_space import SobieskiDesignSpace
from gemseo.scenarios.evaluation import EvaluationScenario
from gemseo.settings.doe import PYDOE_LHS_Settings
from gemseo.utils.platform import PLATFORM_IS_WINDOWS

# %%
# ### 1. Generate your DOE scenario
disciplines = create_discipline([
    "SobieskiPropulsion",
    "SobieskiAerodynamics",
    "SobieskiMission",
    "SobieskiStructure",
])
design_space = SobieskiDesignSpace()
scenario = EvaluationScenario(
    disciplines,
    design_space,
    formulation_settings=MDF_Settings(),
)
scenario.add_observable("y_4")

# %%
#
# !!! note
#     This how-to also works with [MDOScenario][gemseo.scenarios.mdo.MDOScenario],
#     when using Design of Experiments.
#
# ### 2. Execute the scenario with multiprocessing
#
# It is possible to run a DOE in parallel using multiprocessing, in order to do
# this, we specify the number of processes to be used for the computation of
# the samples.
#
# !!! warning
#       The multiprocessing option has some limitations on Windows,
#       so we deactivate it in case of Windows usage.
n_processes = 4 if not PLATFORM_IS_WINDOWS else 1

# %%
# The number of processes is specified through the DOE algorithm settings.
# Although it is an LHS algorithm here,
# it can be specified with any type of DOE algorithm.
lhs_settings = PYDOE_LHS_Settings(
    n_samples=30,
    # Run in parallel on 1 or 4 processors
    n_processes=n_processes,
)

scenario.execute(lhs_settings)

# %%
# !!! warning
#       On Windows, the progress bar may show duplicated instances during the
#       initialization of each subprocess. In some cases it may also print the
#       conclusion of an iteration ahead of another one that was concluded first.
#       This is a consequence of the pickling process and does not affect the
#       computations of the scenario.
#
# ## Summary
#
# The number of processors used by the DOE algorithm can be specified
# in the DOE algorithm settings with the `n_processes` option.
