# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License version 3 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""Uncertainty quantification and management.

The package **uncertainty** provides several functionalities
to quantify and manage uncertainties.
Most of them can be used from the dedicated API.

The sub-package **distributions** offers an abstract level for probability distributions
as well as interfaces to the OpenTURNS and SciPy ones.
It is also possible to fit a probability distribution from data
or select the most likely one from a list of candidates.
These distributions can be used to define random variables in a :class:`.ParameterSpace`
before propagating these uncertainties through a system of :class:`.MDODiscipline`,
by means of a :class:`.DOEScenario`.

.. seealso::

    :class:`.OTDistribution`
    :class:`.SPDistribution`
    :class:`.OTDistributionFitter`

The sub-package **sensitivity** offers an abstract level for sensitivity analysis,
as well as concrete features.
These sensitivity analyses compute indices by means of various methods:
correlation measures, Morris technique and Sobol' variance decomposition.
This sub-package is based in particular on OpenTURNS.

.. seealso::

    :class:`.CorrelationAnalysis`
    :class:`.MorrisAnalysis`
    :class:`.SobolAnalysis`

The sub-package **statistics** offers an abstract level for statistics,
as well as parametric and empirical versions.
Empirical statistics are estimated from a :class:`.Dataset`
while parametric statistics are analytical properties of a :class:`.Distribution`
fitted from a :class:`.Dataset`.

.. seealso::

    :class:`.EmpiricalStatistics`
    :class:`.ParametricStatistics`
"""
from __future__ import annotations
