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
"""# Sort inputs by influence

## Problem

After computing sensitivity indices,
you want to rank the input variables from most to least influential
with respect to a given output.

## Solution

[sort_input_variables()][gemseo.uncertainty.sensitivity.base.BaseSensitivityAnalysis.sort_input_variables]
returns the input variable names ordered by decreasing absolute sensitivity index.
The ranking criterion is the current
[main_method][gemseo.uncertainty.sensitivity.base.BaseSensitivityAnalysis.main_method]
(see [Change the main sensitivity method][]).

## Step-by-step guide
"""

from __future__ import annotations

from gemseo.problems.uncertainty.ishigami.ishigami_discipline import IshigamiDiscipline
from gemseo.problems.uncertainty.ishigami.ishigami_space import IshigamiSpace
from gemseo.uncertainty.sensitivity.correlation import CorrelationAnalysis

# %%
# ### 1. Compute sensitivity indices
#
# Create a
# [CorrelationAnalysis][gemseo.uncertainty.sensitivity.correlation.CorrelationAnalysis]
# on the Ishigami problem
# (see [Compute sensitivity indices][] for the detailed workflow):
discipline = IshigamiDiscipline()
uncertain_space = IshigamiSpace()
analysis = CorrelationAnalysis()
analysis.compute_samples([discipline], uncertain_space, n_samples=1000)
analysis.compute_indices()

# %%
# ### 2. Sort inputs by influence
#
# Pass the output variable name to
# [sort_input_variables()][gemseo.uncertainty.sensitivity.base.BaseSensitivityAnalysis.sort_input_variables]:
analysis.sort_input_variables("y")

# %%
# The ranking uses the default main method (`"Spearman"` for
# [CorrelationAnalysis][gemseo.uncertainty.sensitivity.correlation.CorrelationAnalysis]).
# To rank by a different index type, change
# [main_method][gemseo.uncertainty.sensitivity.base.BaseSensitivityAnalysis.main_method]
# first (see [Change the main sensitivity method][]).

# %%
# ## Summary
#
# - [sort_input_variables(output)][gemseo.uncertainty.sensitivity.base.BaseSensitivityAnalysis.sort_input_variables]
#   returns inputs ranked by decreasing absolute index value;
# - ranking criterion is the current
#   [main_method][gemseo.uncertainty.sensitivity.base.BaseSensitivityAnalysis.main_method];
# - change
#   [main_method][gemseo.uncertainty.sensitivity.base.BaseSensitivityAnalysis.main_method]
#   to rank by a different index type: see [Change the main sensitivity method][].
