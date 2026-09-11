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
"""The Aerostructure MDO test case."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import Final

from gemseo.util.package_import import install_lazy_reexport

if TYPE_CHECKING:
    from collections.abc import Mapping

    # static visibility for mypy / IDEs
    from gemseo.problem.mdo.aerostructure.aerostructure import Aerodynamics  # noqa: F401
    from gemseo.problem.mdo.aerostructure.aerostructure import Mission  # noqa: F401
    from gemseo.problem.mdo.aerostructure.aerostructure import Structure  # noqa: F401
    from gemseo.problem.mdo.aerostructure.aerostructure_design_space import (
        AerostructureDesignSpace,  # noqa: F401
    )

# Class name -> defining submodule (lazy-loaded on attribute access).
_name_to_location: Final[Mapping[str, str]] = MappingProxyType({
    "Aerodynamics": "aerostructure",
    "AerostructureDesignSpace": "aerostructure_design_space",
    "Mission": "aerostructure",
    "Structure": "aerostructure",
})

install_lazy_reexport(globals(), _name_to_location)
