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
"""# Export sensitivity indices to a dataset

## Problem

After computing sensitivity indices,
you want to export them to a
[Dataset][gemseo.dataset.dataset.Dataset]
for further numerical processing or comparison across analyses.

## Solution

[to_dataset()][gemseo.uncertainty.sensitivity.core.base.BaseSensitivityAnalysis.to_dataset]
converts the
[indices][gemseo.uncertainty.sensitivity.core.base.BaseSensitivityAnalysis.indices]
mapping into a
[Dataset][gemseo.dataset.dataset.Dataset]
where inputs and columns are index types for the different outputs.

## Step-by-step guide
"""

from __future__ import annotations

from gemseo.problem.uncertainty.ishigami import IshigamiDiscipline
from gemseo.problem.uncertainty.ishigami import IshigamiSpace
from gemseo.uncertainty.sensitivity import CorrelationAnalysis

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
# ### 2. Export to a dataset
#
# [to_dataset()][gemseo.uncertainty.sensitivity.core.base.BaseSensitivityAnalysis.to_dataset]
# returns a
# [Dataset][gemseo.dataset.dataset.Dataset]
# with all index types:
dataset = analysis.to_dataset()
dataset

# %%
# ## Summary
#
#   [to_dataset()][gemseo.uncertainty.sensitivity.core.base.BaseSensitivityAnalysis.to_dataset]
#   converts the indices dict to a
#   [Dataset][gemseo.dataset.dataset.Dataset].
#
# !!! warning
#     For
#     [SobolAnalysis][gemseo.uncertainty.sensitivity.sobol.SobolAnalysis],
#     second-order indices (one per pair of inputs) are not tabular and land in
#     `dataset.misc` rather than the main table.
