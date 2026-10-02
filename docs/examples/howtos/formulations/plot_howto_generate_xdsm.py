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
"""# Generate an XDSM chart

## Problem

You want to visualize the workflow that corresponds to your scenario,
depending on the [MDO formulation][concept-mdo-formulations].

## Solution

GEMSEO can generate the XDSM of your scenario with the
[xdsmize()][gemseo.scenario.mdo.MDOScenario.xdsmize] method.

## Step-by-step guide
"""

from __future__ import annotations

from gemseo.problem.mdo.sobieski import SobieskiAerodynamics
from gemseo.problem.mdo.sobieski import SobieskiDesignSpace
from gemseo.problem.mdo.sobieski import SobieskiMission
from gemseo.problem.mdo.sobieski import SobieskiPropulsion
from gemseo.problem.mdo.sobieski import SobieskiStructure
from gemseo.scenario import MDOScenario

# %%
# ### Prerequisites
#
# This how-to needs a scenario.
#
# Here, the Sobieski test case is used, with the MDF formulation.

disciplines = [
    SobieskiPropulsion(),
    SobieskiAerodynamics(),
    SobieskiMission(),
    SobieskiStructure(),
]
design_space = SobieskiDesignSpace()
scenario = MDOScenario(disciplines, design_space)
scenario.add_objective("y_4", minimize=False)

# %%
# ### 1. Generate your XDSM
#
# An [MDOScenario][gemseo.scenario.mdo.MDOScenario]
# can generate its own XDSM.
# Several other options can be provided such as pdf generation for instance
scenario.xdsmize(show_html=True, save_html=False)

# %%
# ## Summary
#
# The XDSM of your scenario can be generated with the
# [xdsmize()][gemseo.scenario.mdo.MDOScenario.xdsmize] method.
