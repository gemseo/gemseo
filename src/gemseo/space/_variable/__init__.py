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
"""The hierarchy of variables and its factory."""

from __future__ import annotations

from typing import Final

from gemseo.space._variable._base import BaseVariable
from gemseo.space._variable._base import BoundArray
from gemseo.space._variable._base import BoundType
from gemseo.space._variable._base import ComponentDType
from gemseo.space._variable._base import DataType
from gemseo.space._variable._base import format_components
from gemseo.space._variable._continuous import ContinuousVariable
from gemseo.space._variable._factory import VariableFactory
from gemseo.space._variable._integer import IntegerVariable
from gemseo.space._variable._legacy import Variable

TYPE_MAP: Final[dict[str, ComponentDType]] = {
    cls.model_fields["type"].default: cls.component_type
    for cls in (ContinuousVariable, IntegerVariable)
}
"""The map from a variable data type to the NumPy type of its components."""

__all__ = [
    "TYPE_MAP",
    "BaseVariable",
    "BoundArray",
    "BoundType",
    "ContinuousVariable",
    "DataType",
    "IntegerVariable",
    "Variable",
    "VariableFactory",
    "format_components",
]
