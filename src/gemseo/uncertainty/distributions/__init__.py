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

- an abstract class :class:`.Distribution`
  to define the concept of probability distribution,
- an abstract class :class:`.ComposedDistribution`
  to define the concept of joint probability distribution
  by composing several instances of :class:`.Distribution`,
- a factory :class:`.DistributionFactory` to create instances of :class:`.Distribution`,
- concrete classes implementing these abstracts concepts, by interfacing:

  - the OpenTURNS library:
    :class:`.OTDistribution` and :class:`.OTComposedDistribution`,
  - the Scipy library:
    :class:`.SPDistribution` and :class:`.SPComposedDistribution`.

Lastly, the class :class:`.OTDistributionFitter` offers the possibility
to fit an :class:`.OTDistribution` from data based on OpenTURNS.
"""
from __future__ import annotations
