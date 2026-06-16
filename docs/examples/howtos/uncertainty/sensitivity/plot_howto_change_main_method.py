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
"""# Change the main sensitivity method

## Problem

Each sensitivity analysis computes several index types.
You want to control which index type is used
as the ranking and visualization criterion.

## Solution

The
[main_method][gemseo.uncertainty.sensitivity.base.BaseSensitivityAnalysis.main_method]
attribute sets the active index type.
Its default value is the class attribute `_DEFAULT_MAIN_METHOD`.
It can be reassigned at any point — before or after
[compute_indices()][gemseo.uncertainty.sensitivity.base.BaseSensitivityAnalysis.compute_indices] —
without re-running the analysis.

## Step-by-step guide
"""

from __future__ import annotations

from gemseo.problems.uncertainty.ishigami.ishigami_discipline import IshigamiDiscipline
from gemseo.problems.uncertainty.ishigami.ishigami_space import IshigamiSpace
from gemseo.uncertainty.sensitivity.correlation import CorrelationAnalysis

discipline = IshigamiDiscipline()
uncertain_space = IshigamiSpace()

# %%
# ### 1. Check the default main method
#
# After construction,
# [main_method][gemseo.uncertainty.sensitivity.base.BaseSensitivityAnalysis.main_method]
# is initialized to the class default:
analysis = CorrelationAnalysis()
analysis.main_method

# %%
# ### 2. Compute the indices
#
# The calculation of samples and indices is independent of the main method:
analysis.compute_samples([discipline], uncertain_space, n_samples=1000)
analysis.compute_indices()

# %%
# unlike sorting and visualization methods:
analysis.sort_input_variables("y")

# %%
# ### 3. Change the main method
#
# Assign
# [main_method][gemseo.uncertainty.sensitivity.base.BaseSensitivityAnalysis.main_method]
# directly — no need to recompute indices:
analysis.main_method = "SRC"

# %%
# The ranking now uses the Standardized Regression Coefficient:
analysis.sort_input_variables("y")

# %%
# ## Summary
#
# [main_method][gemseo.uncertainty.sensitivity.base.BaseSensitivityAnalysis.main_method]
# controls which index type is used for ranking and visualization;
#
