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
"""The variables."""

from __future__ import annotations

from typing import Final

# A design space pickled by a release predating the hierarchy of variables
# refers to this class by the name of this package,
# so the name must keep resolving here;
# it is re-exported explicitly and left out of the public surface.
from gemseo.space.variable._legacy import Variable as Variable
from gemseo.space.variable.base import BaseVariable
from gemseo.space.variable.base import ComponentDType
from gemseo.space.variable.base import DataType
from gemseo.space.variable.continuous import ContinuousVariable
from gemseo.space.variable.discrete import DiscreteVariable
from gemseo.space.variable.integer import IntegerVariable

TYPE_MAP: Final[dict[str, ComponentDType]] = {
    cls.type: cls.component_type
    for cls in (ContinuousVariable, DiscreteVariable, IntegerVariable)
}
"""The map from a variable data type to the NumPy type of its components."""

__all__ = [
    "TYPE_MAP",
    "BaseVariable",
    "ContinuousVariable",
    "DataType",
    "DiscreteVariable",
    "IntegerVariable",
]
