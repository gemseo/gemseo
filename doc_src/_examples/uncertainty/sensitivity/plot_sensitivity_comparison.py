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
Comparing sensitivity indices
=============================
"""
from __future__ import annotations

from gemseo.algos.parameter_space import ParameterSpace
from gemseo.api import create_discipline
from gemseo.uncertainty.sensitivity.correlation.analysis import CorrelationAnalysis
from gemseo.uncertainty.sensitivity.morris.analysis import MorrisAnalysis
from matplotlib import pyplot as plt
from numpy import pi

#######################################################################################
# In this example,
# we consider the Ishigami function:
#
# .. math::
#
#    Y=\sin(X_1)+7\sin(X_2)^2+0.1*X_3^4\sin(X_1)
#
# which is well-known in the uncertainty domain:
expressions = {"y": "sin(x1)+7*sin(x2)**2+0.1*x3**4*sin(x1)"}
discipline = create_discipline(
    "AnalyticDiscipline", expressions=expressions, name="Ishigami"
)
#######################################################################################
# The different uncertain variables :math:`X_1` , :math:`X_2` and :math:`X_3`
# are independent and identically distributed
# according to an uniform distribution between :math:`-\pi` and :math:`\pi`:
space = ParameterSpace()
for variable in ["x1", "x2", "x3"]:
    space.add_random_variable(
        variable, "OTUniformDistribution", minimum=-pi, maximum=pi
    )

#######################################################################################
# We would like to carry out two sensitivity analyses,
# e.g. a first one based on correlation coefficients
# and a second one based on the Morris methodology,
# and compare the results,
#
# Firstly,
# we create a :class:`.CorrelationAnalysis`
# and compute the sensitivity indices:
correlation = CorrelationAnalysis([discipline], space, 10)
correlation.compute_indices()

#######################################################################################
# Then,
# we create a :class:`.MorrisAnalysis`
# and compute the sensitivity indices:
morris = MorrisAnalysis([discipline], space, 10)
morris.compute_indices()

#######################################################################################
# Lastly,
# we compare these analyses
# with the graphical method :meth:`.SensitivityAnalysis.plot_comparison`,
# either using a bar chart:
morris.plot_comparison(correlation, "y", use_bar_plot=True, save=False, show=False)

#######################################################################################
# or a radar plot:
morris.plot_comparison(correlation, "y", use_bar_plot=False, save=False, show=False)
# Workaround for HTML rendering, instead of ``show=True``
plt.show()
