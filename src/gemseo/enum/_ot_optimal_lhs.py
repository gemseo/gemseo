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
"""The options of the optimal LHS algorithm from the OpenTURNS library."""

from __future__ import annotations

from enum import StrEnum

# The enumerations below are defined here rather than nested in OTOptimalLHS so that
# the modules using them, e.g. gemseo.doe.openturns.settings.ot_opt_lhs, do not
# import that algorithm and OpenTURNS just for the enumerations.


class TemperatureProfile(StrEnum):
    """The name of the temperature profile."""

    GEOMETRIC = "Geometric"
    LINEAR = "Linear"


class SpaceFillingCriterion(StrEnum):
    """The name of the space-filling criterion."""

    C2 = "C2"
    PHIP = "PhiP"
    MINDIST = "MinDist"
