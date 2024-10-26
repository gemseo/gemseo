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
"""Settings for the SciPy ODE solver algorithms."""

from __future__ import annotations

from numpy import inf
from pydantic import Field
from pydantic import PositiveFloat  # noqa: TCH002

from gemseo.algos.ode.base_ode_solver_settings import BaseODESolverSettings
from gemseo.utils.pydantic_ndarray import NDArrayPydantic  # noqa: TCH001


class BaseScipyODEAlgosSettings(BaseODESolverSettings):
    """Settings for the SciPy ODE solver algorithms."""

    first_step: PositiveFloat | None = Field(
        default=None,
        description=(
            """The initial step size.

            If ``None``, let the algorithm choose.
            """
        ),
    )

    max_step: PositiveFloat = Field(
        default=inf,
        description="""The maximum allowed step size.""",
    )

    rtol: PositiveFloat | NDArrayPydantic[PositiveFloat] = Field(
        default=1e-3,
        description="""The relative tolerance.""",
    )

    atol: PositiveFloat | NDArrayPydantic[PositiveFloat] = Field(
        default=1e-6,
        description="""The absolute tolerance.""",
    )

    jac_sparsity: NDArrayPydantic[float] | None = Field(
        default=None,
        description="""Sparsity structure of the Jacobian matrix.""",
    )
