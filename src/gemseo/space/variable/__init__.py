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

from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import Final

# A design space pickled by a release predating the hierarchy of variables
# refers to this class by the name of this package,
# so the name must keep resolving here;
# it is re-exported explicitly and left out of the public surface.
from gemseo.space.variable._legacy import Variable as Variable
from gemseo.space.variable.base import BaseVariable as BaseVariable
from gemseo.space.variable.base import DataType as DataType  # noqa: TC001
from gemseo.space.variable.deterministic import (
    BaseDeterministicVariable as BaseDeterministicVariable,
)
from gemseo.space.variable.discrete import DiscreteVariable
from gemseo.space.variable.factory import (
    DeterministicVariableFactory as DeterministicVariableFactory,
)
from gemseo.space.variable.integer import IntegerVariable
from gemseo.space.variable.interval import BaseIntervalVariable as BaseIntervalVariable
from gemseo.space.variable.numeric import BaseNumericVariable as BaseNumericVariable
from gemseo.space.variable.numeric import BoundArray as BoundArray
from gemseo.space.variable.numeric import BoundType as BoundType
from gemseo.space.variable.real import RealVariable
from gemseo.util.package_import import install_lazy_reexport

if TYPE_CHECKING:
    from collections.abc import Mapping

    # static visibility for mypy / IDEs
    from gemseo.space.variable.numeric import ComponentDType
    from gemseo.space.variable.random import RandomVariable  # noqa: F401

data_type_to_numpy_type: Final[Mapping[DataType, ComponentDType]] = MappingProxyType({
    cls.type: cls.component_type
    for cls in (DiscreteVariable, IntegerVariable, RealVariable)
})
"""The map from a variable data type to the NumPy type of its components.

Only a variable whose components are numbers has such a type.
"""

# A random variable drags in the probability distributions,
# which a design space has no use for,
# so it is re-exported lazily.
_name_to_location: Final[Mapping[str, str]] = MappingProxyType({
    "RandomVariable": "random"
})

install_lazy_reexport(
    globals(),
    _name_to_location,
    extra_all=[
        "data_type_to_numpy_type",
        "BaseDeterministicVariable",
        "BaseIntervalVariable",
        "BaseNumericVariable",
        "BaseVariable",
        "BoundArray",
        "BoundType",
        "DataType",
        "DeterministicVariableFactory",
        "DiscreteVariable",
        "IntegerVariable",
        "RealVariable",
    ],
)
