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
"""Settings for the SciPy ODE solvers using Jacobian with ``jac_sparsity``."""

from __future__ import annotations

from pydantic import Field

from gemseo.algos.ode.scipy_ode.settings.base_scipy_ode_jac_settings import (
    BaseScipyODESolverJacSettings,
)
from gemseo.utils.pydantic_ndarray import NDArrayPydantic  # noqa: TC001


class BaseScipyODESolverJacSparsitySettings(BaseScipyODESolverJacSettings):
    """Settings for the SciPy ODE solvers using Jacobian and ``jac_sparsity``."""

    jac_sparsity: NDArrayPydantic[float] | None = Field(
        default=None,
        description="""Sparsity structure of the Jacobian matrix.""",
    )
