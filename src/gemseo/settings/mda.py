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
"""Settings of the MDA algorithms."""

from __future__ import annotations

from gemseo.mda.gauss_seidel_settings import MDAGaussSeidel_Settings  # noqa: F401
from gemseo.mda.gs_newton_settings import MDAGSNewton_Settings  # noqa: F401
from gemseo.mda.jacobi_settings import MDAJacobi_Settings  # noqa: F401
from gemseo.mda.mda_chain_settings import MDAChain_Settings  # noqa: F401
from gemseo.mda.newton_raphson_settings import MDANewtonRaphson_Settings  # noqa: F401
from gemseo.mda.quasi_newton_settings import MDAQuasiNewton_Settings  # noqa: F401
from gemseo.mda.sequential_mda_settings import MDASequential_Settings  # noqa: F401
