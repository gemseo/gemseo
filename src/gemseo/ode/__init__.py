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
# Contributors:
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Isabelle Santos
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Algorithms to solve ordinary differential equations (ODEs).

Together with the problem they solve
([ODEProblem][gemseo.ode.problem.ODEProblem]).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Final

from gemseo.util.package_import import install_lazy_reexport

if TYPE_CHECKING:
    # static visibility for mypy / IDEs
    from gemseo.ode.problem import ODEProblem  # noqa: F401
    from gemseo.ode.scipy_ode.settings.bdf import BDF_Settings  # noqa: F401
    from gemseo.ode.scipy_ode.settings.dop853 import DOP853_Settings  # noqa: F401
    from gemseo.ode.scipy_ode.settings.lsoda import LSODA_Settings  # noqa: F401
    from gemseo.ode.scipy_ode.settings.radau import Radau_Settings  # noqa: F401
    from gemseo.ode.scipy_ode.settings.rk23 import RK23_Settings  # noqa: F401
    from gemseo.ode.scipy_ode.settings.rk45 import RK45_Settings  # noqa: F401

# Exported name -> "module.path:Attr" (lazy-loaded on attribute access).
# The module path is relative to this package.
_NAME_TO_LOCATION: Final[dict[str, str]] = {
    "BDF_Settings": "scipy_ode.settings.bdf:BDF_Settings",
    "DOP853_Settings": "scipy_ode.settings.dop853:DOP853_Settings",
    "LSODA_Settings": "scipy_ode.settings.lsoda:LSODA_Settings",
    "ODEProblem": "problem:ODEProblem",
    "RK23_Settings": "scipy_ode.settings.rk23:RK23_Settings",
    "RK45_Settings": "scipy_ode.settings.rk45:RK45_Settings",
    "Radau_Settings": "scipy_ode.settings.radau:Radau_Settings",
}

install_lazy_reexport(globals(), _NAME_TO_LOCATION)
