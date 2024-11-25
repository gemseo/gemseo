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
"""Settings for the LSODA ODE solver from Scipy."""

from __future__ import annotations

from pydantic import Field
from pydantic import NonNegativeInt  # noqa: TC002

from gemseo.algos.ode.scipy_ode.settings.base_scipy_ode_jac_settings import (
    BaseScipyODESolverJacSettings,
)


class LSODA_Settings(BaseScipyODESolverJacSettings):  # noqa: N801
    """Settings for the LSODA ODE solver from Scipy."""

    _TARGET_CLASS_NAME = "LSODA"

    lband: int | None = Field(
        default=None,
        description="""The lower boundary of the bandwidth of the Jacobian.

The SciPy documentation does not mention what happens if ``None``.""",
    )

    uband: int | None = Field(
        default=None,
        description="""The upper boundary of the bandwidth of the Jacobian.

The SciPy documentation does not mention what happens if ``None``.""",
    )

    min_step: NonNegativeInt = Field(
        default=0,
        description="""Minimum allowed step size.""",
    )
