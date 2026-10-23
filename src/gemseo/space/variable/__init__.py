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
from gemseo.space.variable.catalog import CatalogVariable as CatalogVariable
from gemseo.space.variable.categorical import CategoricalVariable as CategoricalVariable
from gemseo.space.variable.discrete import DiscreteVariable as DiscreteVariable
from gemseo.space.variable.integer import IntegerVariable as IntegerVariable
from gemseo.space.variable.real import RealVariable as RealVariable
from gemseo.util.package_import import install_lazy_reexport

if TYPE_CHECKING:
    from collections.abc import Mapping

    # static visibility for mypy / IDEs
    from gemseo.space.variable.random import RandomVariable  # noqa: F401

# A random variable drags in the probability distributions,
# which a design space has no use for,
# so it is re-exported lazily to keep them out of the import of gemseo.space.design;
# the discovery of the deterministic variable factory,
# run by the first call to DesignSpace.add_variable, still imports them.
_name_to_location: Final[Mapping[str, str]] = MappingProxyType({
    "RandomVariable": "random"
})

install_lazy_reexport(
    globals(),
    _name_to_location,
    extra_all=[
        "CatalogVariable",
        "CategoricalVariable",
        "DiscreteVariable",
        "IntegerVariable",
        "RealVariable",
    ],
)
