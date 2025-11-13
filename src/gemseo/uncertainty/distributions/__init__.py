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
"""Capabilities to create and manipulate probability distributions.

This package contains:

- an abstract class
  [BaseDistribution][gemseo.uncertainty.distributions.base_distribution.BaseDistribution]
  to define the concept of probability distribution,
- an abstract class
  [BaseJointDistribution][gemseo.uncertainty.distributions.base_joint.BaseJointDistribution]
  to define the concept of joint probability distribution
  by composing several instances of
  [BaseDistribution][gemseo.uncertainty.distributions.base_distribution.BaseDistribution],
- a factory
  [DistributionFactory][gemseo.uncertainty.distributions.factory.DistributionFactory]
  to create instances of
  [BaseDistribution][gemseo.uncertainty.distributions.base_distribution.BaseDistribution],
- concrete classes implementing these abstracts concepts, by interfacing:

  - the OpenTURNS library:
    [OTDistribution][gemseo.uncertainty.distributions.openturns.distribution.OTDistribution]
    and
    [OTJointDistribution][gemseo.uncertainty.distributions.openturns.joint.OTJointDistribution],
  - the Scipy library:
    [SPDistribution][gemseo.uncertainty.distributions.scipy.distribution.SPDistribution]
    and
    [SPJointDistribution][gemseo.uncertainty.distributions.scipy.joint.SPJointDistribution].

Lastly,
the class
[OTDistributionFitter][gemseo.uncertainty.distributions.openturns.fitting.OTDistributionFitter]
offers the possibility
to fit an
[OTDistribution][gemseo.uncertainty.distributions.openturns.distribution.OTDistribution]
from data based on OpenTURNS.
"""

from __future__ import annotations
