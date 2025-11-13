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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""# Sobol' analysis."""

from __future__ import annotations

from gemseo.problems.uncertainty.ishigami.ishigami_discipline import IshigamiDiscipline
from gemseo.problems.uncertainty.ishigami.ishigami_space import IshigamiSpace
from gemseo.uncertainty.sensitivity.sobol_analysis import SobolAnalysis

# %%
# In this example,
# we consider the Ishigami function
#
# $$ f(x_1,x_2,x_3)=\sin(x_1)+7\sin(x_2)^2+0.1x_3^4\sin(x_1) $$
#
# implemented as an [Discipline][gemseo.core.discipline.discipline.Discipline] by the [IshigamiDiscipline][gemseo.problems.uncertainty.ishigami.ishigami_discipline.IshigamiDiscipline].
# It is commonly used
# with the independent random variables $X_1$, $X_2$ and $X_3$
# uniformly distributed between $-\pi$ and $\pi$
# and defined in the [IshigamiSpace][gemseo.problems.uncertainty.ishigami.ishigami_space.IshigamiSpace].
#
# !!!quote "References"
#       T. Ishigami and T. Homma.
#       An importance quantification technique in uncertainty analysis for computer models.
#       In First International Symposium on Uncertainty Modeling and Analysis. 1990.
discipline = IshigamiDiscipline()
uncertain_space = IshigamiSpace()

# %%
# Then,
# we run sensitivity analysis of type [SobolAnalysis][gemseo.uncertainty.sensitivity.sobol_analysis.SobolAnalysis]:
sensitivity_analysis = SobolAnalysis()
sensitivity_analysis.compute_samples([discipline], uncertain_space, 10000)
sensitivity_analysis.main_method = "total"
sensitivity_analysis.compute_indices()

# %%
# The resulting indices are the first-order, second-order and total-order Sobol' indices:
sensitivity_analysis.indices

# %%
# They can also be accessed separately:

# %%
# One can also obtain their confidence intervals:

# %%
# The main indices are the total Sobol' indices
# ([main_method][gemseo.uncertainty.sensitivity.sobol_analysis.SobolAnalysis.main_method] can also be set to `"first"`
# to use the first-order indices as main indices):

# %%
# These main indices can be used to get the input parameters
# sorted by decreasing order of influence:
sensitivity_analysis.sort_input_variables("y")

# %%
# We can use the method [plot()][gemseo.uncertainty.sensitivity.sobol_analysis.SobolAnalysis.plot]
# to visualize both first-order and total Sobol' indices:
sensitivity_analysis.plot("y", save=False, show=True)

# %%
# Lastly,
# the sensitivity indices can be exported to a [Dataset][gemseo.datasets.dataset.Dataset]:
dataset = sensitivity_analysis.to_dataset()
dataset

# %%
# Note that this view does not contain the second-order Sobol' indices
# as the latter do not have a tabular structure.
# Use the attribute [Dataset.misc][gemseo.datasets.dataset.Dataset.misc] to access it.
