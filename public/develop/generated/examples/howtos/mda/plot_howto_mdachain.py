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
#        :author: Charlie Vanaret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""# Manage nested coupling systems

## Problem

You want to perform a Multidisciplinary Design Analysis (MDA) on a set of coupled disciplines, with a coupling structure made up of both strongly and weakly coupled subsets.
You want the improve the solution performance by taking into account your specific coupling structure.

## Solution

You can create an [MDAChain][gemseo.mda.chain.MDAChain], which will automatically identify the strongly coupled subsets of disciplines (if any) and solve them sequentially. Formally, the MDAChain splits the initial coupled system into weakly coupled subsystems of smaller size, which greatly improves the performance.

## Step-by-step guide
"""

from __future__ import annotations

from gemseo import create_discipline
from gemseo import create_mda
from gemseo.mda.chain_settings import MDAChain_Settings
from gemseo.mda.gauss_seidel_settings import MDAGaussSeidel_Settings

# %%
# ### 1. Create your disciplines
#
# Here, we take the Sobieski disciplines.
# The *Structure*, *Propulsion* and *Aerodynamics* disciplines are highly coupled.
# The *Mission* discipline is weakly couple with the others.

disciplines = create_discipline([
    "SobieskiStructure",
    "SobieskiPropulsion",
    "SobieskiAerodynamics",
    "SobieskiMission",
])

# %%
# ### 2. Create the MDA chain
#
# You can specify which MDA algorithm you want to use inside the chain.
# Here, we set the Gauss-Seidel algorithm.
# Other settings can be provided.
mda = create_mda(
    "MDAChain",
    disciplines,
    settings_model=MDAChain_Settings(inner_mda_settings=MDAGaussSeidel_Settings()),
)

# %%
# ### 3. Observe the coupling structure
#
# The `MDAChain` has decomposed the problem in two parts:
#
# - The `MDAGaussSeidel`, containing the 3 strongly coupled disciplines;
# - The *Mission* discipline, wich is weakly coupled with the 3 other disciplines
for i, element in enumerate(mda.coupling_structure.sequence):
    print(f"Element {i}: {element}", end="\n\n")

# %%
# You can also access to the inner MDAs,
# e.g. the first one:
mda_gauss_seidel = mda.inner_mdas[0]
mda_gauss_seidel.disciplines

# %%
# We can see that GEMSEO automatically excludes the *Mission* discipline in the MDA,
# because this discipline is not strongly coupled.
#
# ## Summary
#
# The [MDAChain][gemseo.mda.chain.MDAChain] allows you
# to consider a huge amount of disciplines.
# The workflow is automatically built,
# splitting the groups of disciplines that have to be executed together into sub-MDAs.
#
# The chain gathers the sub-MDAs in its
# [inner_mdas][gemseo.mda.chain.MDAChain.inner_mdas] attribute.
