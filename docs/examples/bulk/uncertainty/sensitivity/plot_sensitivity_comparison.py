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
"""# Comparing sensitivity indices."""

from __future__ import annotations

from gemseo.problems.uncertainty.ishigami.ishigami_discipline import IshigamiDiscipline
from gemseo.problems.uncertainty.ishigami.ishigami_space import IshigamiSpace
from gemseo.uncertainty.sensitivity.correlation_analysis import CorrelationAnalysis
from gemseo.uncertainty.sensitivity.morris_analysis import MorrisAnalysis

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
# We would like to carry out two sensitivity analyses,
# e.g. a first one based on correlation coefficients
# and a second one based on the Morris methodology,
# and compare the results,
#
# Firstly,
# we create a [CorrelationAnalysis][gemseo.uncertainty.sensitivity.correlation_analysis.CorrelationAnalysis] and compute the sensitivity indices:
correlation = CorrelationAnalysis()
correlation.compute_samples([discipline], uncertain_space, 10)
correlation.compute_indices()

# %%
# Then,
# we create an [MorrisAnalysis][gemseo.uncertainty.sensitivity.morris_analysis.MorrisAnalysis] and compute the sensitivity indices:
morris = MorrisAnalysis()
morris.compute_samples([discipline], uncertain_space, 10)
morris.compute_indices()

# %%
# Lastly,
# we compare these analyses
# with the graphical method [plot_comparison()][gemseo.uncertainty.sensitivity.base_sensitivity_analysis.BaseSensitivityAnalysis.plot_comparison],
# either using a bar chart:
morris.plot_comparison(correlation, "y", use_bar_plot=True, save=False, show=True)

# %%
# or a radar plot:
morris.plot_comparison(correlation, "y", use_bar_plot=False, save=False, show=True)

# %%
# !!! tip
#
#     The comparison method can use plotly to generate an interactive web-based figure.
#     Just set the execution option `file_format` to `"html"`.
