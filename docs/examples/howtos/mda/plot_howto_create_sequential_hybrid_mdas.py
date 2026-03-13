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
"""# Create sequential hybrid MDAs

## Problem

You want to solve couplings by using different types of MDAs sequentially.
For instance,
you first want to execute the Jacobi algorithm in parallel,
and then use a Gauss-Seidel approach to ensure the convergence.

## Solution

You need to create an [MDASequential][gemseo.mda.sequential.MDASequential].

## Step-by-step guide
"""

from __future__ import annotations

from gemseo import create_discipline
from gemseo import create_mda
from gemseo.mda.jacobi_settings import MDAJacobi_Settings

# %%
# ### 1. Create your disciplines
#
# Here, we take the Sobieski disciplines.
# The *Structure*, *Propulsion* and *Aerodynamics* disciplines are strongly coupled.
# The *Mission* discipline is weakly coupled with the three others.

disciplines = create_discipline([
    "SobieskiStructure",
    "SobieskiPropulsion",
    "SobieskiAerodynamics",
    "SobieskiMission",
])

# %%
# ### 2. Create the different MDAs
#
# We want at most 5 iterations of Jacobi,
# and then use Gauss-Seidel.
mda1 = create_mda(
    "MDAJacobi", disciplines, settings_model=MDAJacobi_Settings(max_mda_iter=5)
)
mda2 = create_mda("MDAGaussSeidel", disciplines)

# %%
# ### 3. Create the sequential MDA, and execute it
mda = create_mda(
    "MDASequential",
    disciplines,
    mda_sequence=[mda1, mda2],
)
res = mda.execute()

# %%
# ### 4. Analyze the result through the residuals
#
# We can clearly see the 5 first iterations through an MDA (here, Jacobi),
# and then executing another MDA (Gauss-Seidel).
mda.plot_residual_history(logscale=(1e-8, 10.0), save=False, show=True)

# %%
# ## Summary
#
# You can hybrid different MDAs by creating an
# [MDASequential][gemseo.mda.sequential.MDASequential].
# In that case, you have to give the sequence of MDAs.
