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
"""Scipy-based capabilities for probability distributions.

This package interfaces capabilities
from the [SciPy](https://scipy.org/) library.

## Interfaced distributions

This package implements the abstract classes
[BaseDistribution][gemseo.uncertainty.distribution.core.base.BaseDistribution]
and
[BaseJointDistribution][gemseo.uncertainty.distribution.core.base_joint.BaseJointDistribution].

## Classical distributions

This module also implements a deliberately limited selection
of classical probability distributions
in a user-friendly way:
[SPExponentialDistribution][gemseo.uncertainty.distribution.scipy.exponential.SPExponentialDistribution],
[SPNormalDistribution][gemseo.uncertainty.distribution.scipy.normal.SPNormalDistribution],
[SPTriangularDistribution][gemseo.uncertainty.distribution.scipy.triangular.SPTriangularDistribution],
and
[SPUniformDistribution][gemseo.uncertainty.distribution.scipy.uniform.SPUniformDistribution].
More precisely,
the argument whose nature is a dictionary of keyword parameters
is replaced with several user-defined keyword arguments.
In this way, the use writes `SPUniformDistribution('x', -1., 3.)`
or `SPUniformDistribution('x', minimum=-1., maximum=3.)`
instead of  `SPDistribution('x', 'Uniform', {"loc": -1, "scale": 4})`.
Furthermore,
these classes inheriting from
[SPDistribution][gemseo.uncertainty.distribution.scipy.distribution.SPDistribution]
are documented in such a way that a newbie could easily apprehend them.

## Joint probability distribution

An
[SPDistribution][gemseo.uncertainty.distribution.scipy.distribution.SPDistribution]
has a
[JOINT_DISTRIBUTION_CLASS][gemseo.uncertainty.distribution.scipy.distribution.SPDistribution.JOINT_DISTRIBUTION_CLASS]
which is a class to build a joint probability distribution
related to given random variables from a list of
[SPDistribution][gemseo.uncertainty.distribution.scipy.distribution.SPDistribution]
objects
implementing the probability distributions of these variables
based on the SciPy library and from a copula name.

Note:
   A copula is a mathematical function used to define the dependence
   between random variables from their cumulative density functions.
   [See more](https://en.wikipedia.org/wiki/Copula_(probability_theory)).
"""

from __future__ import annotations
