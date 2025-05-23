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
"""
Morris analysis
===============
"""

from __future__ import annotations

import pprint

from gemseo import configure_logger
from gemseo.problems.uncertainty.ishigami.ishigami_discipline import IshigamiDiscipline
from gemseo.problems.uncertainty.ishigami.ishigami_space import IshigamiSpace
from gemseo.uncertainty.sensitivity.morris_analysis import MorrisAnalysis

configure_logger()

# %%
# In this example,
# we consider the Ishigami function :cite:`ishigami1990`
#
# .. math::
#
#    f(x_1,x_2,x_3)=\sin(x_1)+7\sin(x_2)^2+0.1x_3^4\sin(x_1)
#
# implemented as an :class:`.Discipline` by the :class:`.IshigamiDiscipline`.
# It is commonly used
# with the independent random variables :math:`X_1`, :math:`X_2` and :math:`X_3`
# uniformly distributed between :math:`-\pi` and :math:`\pi`
# and defined in the :class:`.IshigamiSpace`.

discipline = IshigamiDiscipline()
uncertain_space = IshigamiSpace()

# %%
# Then,
# we run sensitivity analysis of type :class:`.MorrisAnalysis`:
sensitivity_analysis = MorrisAnalysis()
sensitivity_analysis.compute_samples([discipline], uncertain_space, n_samples=0)
sensitivity_analysis.compute_indices()

# %%
# The resulting indices are the empirical means and the standard deviations
# of the absolute output variations due to input changes.
sensitivity_analysis.indices

# %%
# The main indices corresponds to these empirical means
# (this main method can be changed with :attr:`.MorrisAnalysis.main_method`):
pprint.pprint(sensitivity_analysis.main_indices)

# %%
# and can be interpreted with respect to the empirical bounds of the outputs:
pprint.pprint(sensitivity_analysis.outputs_bounds)

# %%
# We can also get the input parameters sorted by decreasing order of influence:
sensitivity_analysis.sort_input_variables("y")

# %%
# We can use the method :meth:`.MorrisAnalysis.plot`
# to visualize the different series of indices:
sensitivity_analysis.plot("y", save=False, show=True, lower_mu=0, lower_sigma=0)

# %%
# Lastly,
# the sensitivity indices can be exported to a :class:`.Dataset`:
sensitivity_analysis.to_dataset()
