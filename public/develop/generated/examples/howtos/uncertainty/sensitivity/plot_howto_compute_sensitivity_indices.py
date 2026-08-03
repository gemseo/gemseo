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
"""# Compute sensitivity indices

## Problem

You want to quantify how much each uncertain input variable contributes
to the variability of a discipline output.

## Solution

Every sensitivity analysis in GEMSEO follows the same two-step workflow:

1. [compute_samples()][gemseo.uncertainty.sensitivity.core.base.BaseSensitivityAnalysis.compute_samples]
   evaluates the discipline over a set of random inputs;
2. [compute_indices()][gemseo.uncertainty.sensitivity.core.base.BaseSensitivityAnalysis.compute_indices]
   derives the sensitivity indices from those samples.

The result is stored in
[indices][gemseo.uncertainty.sensitivity.core.base.BaseSensitivityAnalysis.indices].

## Step-by-step guide
"""

from __future__ import annotations

from gemseo.problem.uncertainty.ishigami import IshigamiDiscipline
from gemseo.problem.uncertainty.ishigami import IshigamiSpace
from gemseo.uncertainty.sensitivity import CorrelationAnalysis

# %%
# ### 1. Set up the test problem
#
# The Ishigami function is a standard benchmark for sensitivity analysis:
#
# $$f(x_1,x_2,x_3)=\sin(x_1)+7\sin(x_2)^2+0.1x_3^4\sin(x_1)$$
#
# with $X_1, X_2, X_3 \sim \mathcal{U}(-\pi, \pi)$ independently.
#
# !!!quote "Reference"
#     T. Ishigami and T. Homma,
#     *An importance quantification technique in uncertainty analysis for computer models*,
#     First International Symposium on Uncertainty Modeling and Analysis, 1990.
discipline = IshigamiDiscipline()
uncertain_space = IshigamiSpace()

# %%
# ### 2. Instantiate a sensitivity analysis
#
# [CorrelationAnalysis][gemseo.uncertainty.sensitivity.correlation.CorrelationAnalysis]
# is used here as a representative example.
# The same workflow applies to every analysis class:
analysis = CorrelationAnalysis()

# %%
# ### 3. Generate samples
#
# [compute_samples()][gemseo.uncertainty.sensitivity.core.base.BaseSensitivityAnalysis.compute_samples]
# evaluates the discipline at random input points drawn from the uncertain space:
samples = analysis.compute_samples([discipline], uncertain_space, n_samples=1000)
samples

# %%
# ### 4. Compute indices
#
# [compute_indices()][gemseo.uncertainty.sensitivity.core.base.BaseSensitivityAnalysis.compute_indices]
# derives the sensitivity indices from the samples:
analysis.compute_indices()

# %%
# ### 5. Inspect the indices
#
# The
# [indices][gemseo.uncertainty.sensitivity.core.base.BaseSensitivityAnalysis.indices]
# attribute holds a nested mapping `{output_name: {index_name: {input_name: value}}}`:
analysis.indices.pearson

# %%
# ## Summary
#
# - Call
#   [compute_samples()][gemseo.uncertainty.sensitivity.core.base.BaseSensitivityAnalysis.compute_samples]
#   to evaluate the discipline over a specific design;
# - Call
#   [compute_indices()][gemseo.uncertainty.sensitivity.core.base.BaseSensitivityAnalysis.compute_indices]
#   to derive the indices from those samples;
# - Access indices via
#   [indices][gemseo.uncertainty.sensitivity.core.base.BaseSensitivityAnalysis.indices].
