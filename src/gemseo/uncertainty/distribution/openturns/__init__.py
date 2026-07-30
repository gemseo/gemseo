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
"""OpenTURNS-based capabilities for probability distributions.

This package interfaces capabilities
from the [OpenTURNS](https://openturns.github.io/www/) library.

## Interfaced distributions

This package implements the abstract classes
[BaseDistribution][gemseo.uncertainty.distribution.core.base.BaseDistribution]
and
[BaseJointDistribution][gemseo.uncertainty.distribution.core.base_joint.BaseJointDistribution].

## Classical distributions

This package also implements a deliberately limited selection
of standard probability distributions
in a user-friendly way:
[OTExponentialDistribution][gemseo.uncertainty.distribution.openturns.exponential.OTExponentialDistribution],
[OTNormalDistribution][gemseo.uncertainty.distribution.openturns.normal.OTNormalDistribution],
[OTTriangularDistribution][gemseo.uncertainty.distribution.openturns.triangular.OTTriangularDistribution],
and
[OTUniformDistribution][gemseo.uncertainty.distribution.openturns.uniform.OTUniformDistribution].
More precisely,
the argument whose nature is a tuple of positional parameters
is replaced with several user-defined keyword arguments.
In this way, the use writes `OTUniformDistribution('x', -1., 3.)`
or `OTUniformDistribution('x', minimum=-1., maximum=3.)`
instead of `OTDistribution('x', 'Uniform', (-1., 3.))`.
Furthermore,
these classes inheriting from
[OTDistribution][gemseo.uncertainty.distribution.openturns.distribution.OTDistribution]
are documented in such a way that a newbie could easily apprehend them.

## Joint probability distribution

An
[OTDistribution][gemseo.uncertainty.distribution.openturns.distribution.OTDistribution]
has a
[JOINT_DISTRIBUTION_CLASS][gemseo.uncertainty.distribution.openturns.distribution.OTDistribution.JOINT_DISTRIBUTION_CLASS]
which is a class to build a joint probability distribution
related to given random variables from a list of
[OTDistribution][gemseo.uncertainty.distribution.openturns.distribution.OTDistribution]
objects
implementing the probability distributions of these variables
based on the OpenTURNS library and from a copula name.

Note:
   A copula is a mathematical function used to define the dependence
   between random variables from their cumulative density functions.
   [See more](https://en.wikipedia.org/wiki/Copula_(probability_theory)).

## Distribution fitting

The class
[OTDistributionFitter][gemseo.uncertainty.distribution.openturns.distribution_fitter.OTDistributionFitter]
 offers the possibility
to fit an
[OTDistribution][gemseo.uncertainty.distribution.openturns.distribution.OTDistribution]
from `numpy.array` data,
based on the OpenTURNS capabilities.
"""

from __future__ import annotations
