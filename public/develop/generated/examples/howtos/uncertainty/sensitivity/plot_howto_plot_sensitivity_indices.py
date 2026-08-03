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
"""# Plot sensitivity indices

## Problem

After computing sensitivity indices,
you want to visualize them as a bar chart or a radar plot.

## Solution

[BaseSensitivityAnalysis][gemseo.uncertainty.sensitivity.core.base.BaseSensitivityAnalysis]
provides two generic visualization methods available for every analysis class:

- [plot_bar()][gemseo.uncertainty.sensitivity.core.base.BaseSensitivityAnalysis.plot_bar]
  draws a horizontal bar chart of the main indices for one output;
- [plot_radar()][gemseo.uncertainty.sensitivity.core.base.BaseSensitivityAnalysis.plot_radar]
  draws a radar (spider) chart of the main indices for one output,
- [plot()][gemseo.uncertainty.sensitivity.core.base.BaseSensitivityAnalysis.plot]
  draws a specific chart for one output;
  this method is not implemented by default.

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
# ### 2. Bar chart
#
# [plot_bar()][gemseo.uncertainty.sensitivity.core.base.BaseSensitivityAnalysis.plot_bar]
# shows the sensitivity of the
# [main_method][gemseo.uncertainty.sensitivity.core.base.BaseSensitivityAnalysis.main_method]
# for each input variable:
analysis.plot_bar("y", save=False, show=True)

# %%
# ### 3. Radar chart
#
# [plot_radar()][gemseo.uncertainty.sensitivity.core.base.BaseSensitivityAnalysis.plot_radar]
# shows the same information on a radar (spider) plot:
analysis.plot_radar("y", save=False, show=True)

# %%
# ### 3. Specific chart
#
# [plot()][gemseo.uncertainty.sensitivity.core.base.BaseSensitivityAnalysis.plot]
# shows the sensitivity indices
# using a chart specific to the type of sensitivity analysis:
analysis.plot("y", save=False, show=True)

# %%
# ## Summary
#
# - [plot_bar(output)][gemseo.uncertainty.sensitivity.core.base.BaseSensitivityAnalysis.plot_bar]
#   draws a bar chart of main indices per input variable;
# - [plot_radar(output)][gemseo.uncertainty.sensitivity.core.base.BaseSensitivityAnalysis.plot_radar]
#   draws a radar chart of main indices per input variable;
# - [plot(output)][gemseo.uncertainty.sensitivity.core.base.BaseSensitivityAnalysis.plot]
#   displays the indices
#   using a chart specific to the type of sensitivity analysis:
# - both use the current
#   [main_method][gemseo.uncertainty.sensitivity.core.base.BaseSensitivityAnalysis.main_method]
#   (see [Change the main sensitivity method][]).
