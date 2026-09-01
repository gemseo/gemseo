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
"""# Compare sensitivity analyses

## Problem

You have run two different sensitivity analyses on the same problem
and want to see whether they agree on which inputs are most influential.

## Solution

[BaseSensitivityAnalysis.plot_comparison()][gemseo.uncertainty.sensitivity.core.base.BaseSensitivityAnalysis.plot_comparison]
overlays the main indices of two analyses in a single figure,
either as a bar chart or as a radar plot.

## Step-by-step guide
"""

from __future__ import annotations

from gemseo.problem.uncertainty.ishigami import IshigamiDiscipline
from gemseo.problem.uncertainty.ishigami import IshigamiSpace
from gemseo.uncertainty.sensitivity import CorrelationAnalysis
from gemseo.uncertainty.sensitivity import MorrisAnalysis

# %%
# ### 1. Set up the test problem
#
# The Ishigami function is a standard benchmark for sensitivity analysis [@ishigami1990]:
#
# $$f(x_1,x_2,x_3)=\sin(x_1)+7\sin(x_2)^2+0.1x_3^4\sin(x_1)$$
#
# with $X_1, X_2, X_3 \sim \mathcal{U}(-\pi, \pi)$ independently.
discipline = IshigamiDiscipline()
uncertain_space = IshigamiSpace()

# %%
# ### 2. Run two sensitivity analyses
#
# [CorrelationAnalysis][gemseo.uncertainty.sensitivity.correlation.CorrelationAnalysis]:
correlation = CorrelationAnalysis()
correlation.compute_samples([discipline], uncertain_space, n_samples=1000)
correlation.compute_indices()

# %%
# [MorrisAnalysis][gemseo.uncertainty.sensitivity.morris.MorrisAnalysis]:
morris = MorrisAnalysis()
morris.compute_samples([discipline], uncertain_space, n_samples=0)
morris.compute_indices()

# %%
# ### 3. Compare with a bar chart
#
# Pass the second analysis to
# [plot_comparison()][gemseo.uncertainty.sensitivity.core.base.BaseSensitivityAnalysis.plot_comparison],
# to compare the methods by normalizing the indices between 0 and 1:
morris.plot_comparison(correlation, "y", save=False, show=True)

# %%
# ### 4. Compare with a radar plot
morris.plot_comparison(correlation, "y", use_bar_plot=False, save=False, show=True)

# %%
# ## Summary
#
# - Run two analyses with matching disciplines and uncertain space;
# - [plot_comparison()][gemseo.uncertainty.sensitivity.core.base.BaseSensitivityAnalysis.plot_comparison]
#   overlays their main indices normalized between 0 and 1.
# - Pass `use_bar_plot=True` for bars, `False` for radar.
