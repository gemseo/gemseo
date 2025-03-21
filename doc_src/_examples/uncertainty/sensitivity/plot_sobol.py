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
Sobol' analysis
===============
"""

from __future__ import annotations

import pprint

from gemseo import configure_logger
from gemseo.problems.uncertainty.ishigami.ishigami_discipline import IshigamiDiscipline
from gemseo.problems.uncertainty.ishigami.ishigami_space import IshigamiSpace
from gemseo.uncertainty.sensitivity.sobol_analysis import SobolAnalysis

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
# we run sensitivity analysis of type :class:`.SobolAnalysis`:
sensitivity_analysis = SobolAnalysis()
sensitivity_analysis.compute_samples([discipline], uncertain_space, 10000)
sensitivity_analysis.main_method = "total"
sensitivity_analysis.compute_indices()

# %%
# The resulting indices are the first-order, second-order and total-order Sobol' indices:
sensitivity_analysis.indices

# %%
# They can also be accessed separately:
pprint.pprint(sensitivity_analysis.indices.first)
pprint.pprint(sensitivity_analysis.indices.second)
pprint.pprint(sensitivity_analysis.indices.total)

# %%
# One can also obtain their confidence intervals:
pprint.pprint(sensitivity_analysis.get_intervals())
pprint.pprint(sensitivity_analysis.get_intervals(first_order=False))

# %%
# The main indices are the total Sobol' indices
# (:attr:`.SobolAnalysis.main_method` can also be set to ``"first"``
# to use the first-order indices as main indices):
pprint.pprint(sensitivity_analysis.main_indices)

# %%
# These main indices can be used to get the input parameters
# sorted by decreasing order of influence:
sensitivity_analysis.sort_input_variables("y")

# %%
# We can use the method :meth:`.SobolAnalysis.plot`
# to visualize both first-order and total Sobol' indices:
sensitivity_analysis.plot("y", save=False, show=True)

# %%
# Lastly,
# the sensitivity indices can be exported to a :class:`.Dataset`:
dataset = sensitivity_analysis.to_dataset()
dataset

# %%
# Note that this view does not contain the second-order Sobol' indices
# as the latter do not have a tabular structure.
# Use the attribute :attr:`.Dataset.misc` to access it.
pprint.pprint(dataset.misc)
