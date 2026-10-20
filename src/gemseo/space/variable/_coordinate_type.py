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
"""The NumPy types of the coordinates of the variables."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import Final

from gemseo.enum import VariableType
from gemseo.space.variable.catalog import CatalogVariable
from gemseo.space.variable.categorical import CategoricalVariable
from gemseo.space.variable.discrete import DiscreteVariable
from gemseo.space.variable.integer import IntegerVariable
from gemseo.space.variable.numeric import BaseNumericVariable
from gemseo.space.variable.real import RealVariable

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gemseo.space.variable.base import CoordinateDType

design_variable_type_to_numpy_type: Final[Mapping[VariableType, CoordinateDType]] = (
    MappingProxyType({
        cls.type: cls.coordinate_type
        for cls in (
            CatalogVariable,
            CategoricalVariable,
            DiscreteVariable,
            IntegerVariable,
            RealVariable,
        )
    })
)
"""The map from a design variable type to the NumPy type of its coordinates.

A design space cannot hold a random variable, so the random variable type is left out.
"""

variable_type_to_numpy_type: Final[Mapping[VariableType, CoordinateDType]] = (
    MappingProxyType({
        **design_variable_type_to_numpy_type,
        # RandomVariable drags in the probability distributions,
        # which a design space has no use for,
        # so its type is not read from the class;
        # it inherits the NumPy type of its coordinates.
        VariableType.RANDOM: BaseNumericVariable.coordinate_type,
    })
)
"""The map from a variable type to the NumPy type of its coordinates."""
