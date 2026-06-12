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
"""# Accelerate MDA convergence

## Problem

Your MDA requires too many iterations (and thus too many execution/linearization of your disciplines)
and you would like to speed-up the convergence.

## Solution

[MDA acceleration methods][concept-mda-acceleration-relaxation]
may decrease the number of needed iterations.

## Step-by-step guide
"""

from __future__ import annotations

from gemseo import create_discipline
from gemseo import create_mda
from gemseo.algos.sequence_transformer.acceleration import AccelerationMethod
from gemseo.mda.gauss_seidel_settings import MDAGaussSeidel_Settings

# %%
# ### 1. Create your disciplines
#
# Let's start creating the well-known Sobieski disciplines.

disciplines = create_discipline([
    "SobieskiStructure",
    "SobieskiPropulsion",
    "SobieskiAerodynamics",
    "SobieskiMission",
])

# %%
# ### 2. Default Gauss-Seidel MDA
gauss_seidel_mda = create_mda(
    "MDAGaussSeidel",
    disciplines,
    settings_model=MDAGaussSeidel_Settings(max_mda_iter=15),
)
gauss_seidel_mda.execute()
gauss_seidel_mda.plot_residual_history(logscale=[1e-8, 10.0], save=False, show=True)

# %%
# The MDA has converged in 8 iterations.
#
# ### 3. Accelerated Gauss-Seidel MDA
accelerated_gauss_seidel_mda = create_mda(
    "MDAGaussSeidel",
    disciplines,
    settings_model=MDAGaussSeidel_Settings(
        max_mda_iter=15, acceleration_method=AccelerationMethod.MINIMUM_POLYNOMIAL
    ),
)
accelerated_gauss_seidel_mda.execute()
accelerated_gauss_seidel_mda.plot_residual_history(
    logscale=[1e-8, 10.0], save=False, show=True
)

# %%
# In our case, the selected acceleration method allowed to save 3 iterations.
# In some cases, the number of iterations can be cut-off more significantly.
#
# ## Summary
#
# Applying acceleration methods may cut-off
# the number of iterations needed by the MDA to converge.
#
# !!! warning
#     These methods use the previous iterates to extrapolate a new iterate, hopefully speeding-up the convergence.
#     In some cases, the new iterate may fall outside the boundaries defined in the grammars.
#     Moreover, in some rare cases,
#     acceleration methods can increase the number of iterations.
