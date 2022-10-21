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

from gemseo.algos.parameter_space import ParameterSpace
from gemseo.api import create_discipline
from gemseo.uncertainty.sensitivity.morris.analysis import MorrisAnalysis
from matplotlib import pyplot as plt
from numpy import pi

#######################################################################################
# In this example,
# we consider a function from :math:`[-\pi,\pi]^3` to :math:`\mathbb{R}^3`:
#
# .. math::
#
#    (y_1,y_2)=\left(f(x_1,x_2,x_3),f(x_2,x_1,x_3)\right)
#
# where :math:`f(a,b,c)=\sin(a)+7\sin(b)^2+0.1*c^4\sin(a)` is the Ishigami function:

expressions = {
    "y1": "sin(x1)+7*sin(x2)**2+0.1*x3**4*sin(x1)",
    "y2": "sin(x2)+7*sin(x1)**2+0.1*x3**4*sin(x2)",
}
discipline = create_discipline(
    "AnalyticDiscipline", expressions=expressions, name="Ishigami2"
)

#######################################################################################
# Then,
# we consider the case where
# the deterministic variables :math:`x_1`, :math:`x_2` and :math:`x_3` are replaced
# with the uncertain variables :math:`X_1`, :math:`X_2` and :math:`X_3`.
# The latter are independent and identically distributed
# according to a uniform distribution between :math:`-\pi` and :math:`\pi`:
space = ParameterSpace()
for variable in ["x1", "x2", "x3"]:
    space.add_random_variable(
        variable, "OTUniformDistribution", minimum=-pi, maximum=pi
    )

#######################################################################################
# From that,
# we would like to carry out a sensitivity analysis with the random outputs
# :math:`Y_1=f(X_1,X_2,X_3)` and :math:`Y_2=f(X_2,X_1,X_3)`.
# For that,
# we can compute the correlation coefficients from a :class:`.MorrisAnalysis`:
morris = MorrisAnalysis([discipline], space, 10)
morris.compute_indices()

#######################################################################################
# The resulting indices are the empirical means and the standard deviations
# of the absolute output variations due to input changes.
pprint.pprint(morris.indices)

#######################################################################################
# The main indices corresponds to these empirical means
# (this main method can be changed with :attr:`.MorrisAnalysis.main_method`):
pprint.pprint(morris.main_indices)

#######################################################################################
# and can be interpreted with respect to the empirical bounds of the outputs:
pprint.pprint(morris.outputs_bounds)

#######################################################################################
# We can also sort the input parameters by decreasing order of influence
# and observe that this ranking is not the same for both outputs:
print(morris.sort_parameters("y1"))
print(morris.sort_parameters("y2"))

#######################################################################################
# Lastly,
# we can use the method :meth:`.MorrisAnalysis.plot`
# to visualize the different series of indices:
morris.plot("y1", save=False, show=False, lower_mu=0, lower_sigma=0)
morris.plot("y2", save=False, show=False, lower_mu=0, lower_sigma=0)
# Workaround for HTML rendering, instead of ``show=True``
plt.show()
