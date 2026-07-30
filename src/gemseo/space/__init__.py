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
"""Design and parameter spaces."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Final

from gemseo.util.package_import import install_lazy_reexport

if TYPE_CHECKING:
    # static visibility for mypy / IDEs
    from gemseo.space.design import DesignSpace  # noqa: F401
    from gemseo.space.factory import DESIGN_SPACE_FACTORY  # noqa: F401
    from gemseo.space.factory import PARAMETER_SPACE_FACTORY  # noqa: F401
    from gemseo.space.parameter import ParameterSpace  # noqa: F401

# Class name -> defining submodule (lazy-loaded on attribute access).
_NAME_TO_LOCATION: Final[dict[str, str]] = {
    "DESIGN_SPACE_FACTORY": "factory",
    "DesignSpace": "design",
    "PARAMETER_SPACE_FACTORY": "factory",
    "ParameterSpace": "parameter",
}

install_lazy_reexport(globals(), _NAME_TO_LOCATION)
