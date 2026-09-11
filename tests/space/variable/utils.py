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
"""Helpers shared by the tests of the variable hierarchy."""

from __future__ import annotations

from gemseo.space.variable import ContinuousVariable
from gemseo.space.variable import DiscreteVariable
from gemseo.space.variable import IntegerVariable

kinds = (ContinuousVariable, IntegerVariable)
"""The kinds of variable whose domain is an interval."""

all_kinds = (*kinds, DiscreteVariable)
"""All the kinds of variable."""

kind_to_kwargs = {
    ContinuousVariable: {"size": 1, "lower_bound": 0, "upper_bound": 1},
    IntegerVariable: {"size": 1, "lower_bound": 0, "upper_bound": 1},
    # A discrete variable derives its bounds from its choices.
    DiscreteVariable: {"choices": [0, 1]},
}
"""The arguments building a variable of size 1 with the bounds [0, 1], per kind."""
