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
"""Multidisciplinary Analyses : coupled system solvers."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Final

from gemseo.util.package_import import install_lazy_reexport

if TYPE_CHECKING:
    # static visibility for mypy / IDEs
    from gemseo.mda.chain import MDAChain  # noqa: F401
    from gemseo.mda.chain_settings import MDAChain_Settings  # noqa: F401
    from gemseo.mda.factory import MDA_FACTORY  # noqa: F401
    from gemseo.mda.gauss_seidel import MDAGaussSeidel  # noqa: F401
    from gemseo.mda.gauss_seidel_newton_raphson import (
        MDAGaussSeidelNewtonRaphson,  # noqa: F401
    )
    from gemseo.mda.gauss_seidel_newton_raphson_settings import (
        MDAGaussSeidelNewtonRaphson_Settings,  # noqa: F401
    )
    from gemseo.mda.gauss_seidel_settings import MDAGaussSeidel_Settings  # noqa: F401
    from gemseo.mda.jacobi import MDAJacobi  # noqa: F401
    from gemseo.mda.jacobi_settings import MDAJacobi_Settings  # noqa: F401
    from gemseo.mda.newton_raphson import MDANewtonRaphson  # noqa: F401
    from gemseo.mda.newton_raphson_settings import (
        MDANewtonRaphson_Settings,  # noqa: F401
    )
    from gemseo.mda.quasi_newton import MDAQuasiNewton  # noqa: F401
    from gemseo.mda.quasi_newton_settings import MDAQuasiNewton_Settings  # noqa: F401
    from gemseo.mda.sequential import MDASequential  # noqa: F401
    from gemseo.mda.sequential_settings import MDASequential_Settings  # noqa: F401

# Class name -> defining submodule (lazy-loaded on attribute access).
_NAME_TO_LOCATION: Final[dict[str, str]] = {
    "MDAChain": "chain",
    "MDAChain_Settings": "chain_settings",
    "MDA_FACTORY": "factory",
    "MDAGaussSeidel": "gauss_seidel",
    "MDAGaussSeidelNewtonRaphson": "gauss_seidel_newton_raphson",
    "MDAGaussSeidelNewtonRaphson_Settings": "gauss_seidel_newton_raphson_settings",
    "MDAGaussSeidel_Settings": "gauss_seidel_settings",
    "MDAJacobi": "jacobi",
    "MDAJacobi_Settings": "jacobi_settings",
    "MDANewtonRaphson": "newton_raphson",
    "MDANewtonRaphson_Settings": "newton_raphson_settings",
    "MDAQuasiNewton": "quasi_newton",
    "MDAQuasiNewton_Settings": "quasi_newton_settings",
    "MDASequential": "sequential",
    "MDASequential_Settings": "sequential_settings",
}

install_lazy_reexport(globals(), _NAME_TO_LOCATION)
