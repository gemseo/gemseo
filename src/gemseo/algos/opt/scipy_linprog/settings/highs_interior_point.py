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
"""Settings for the HiGHS interior point method."""

from __future__ import annotations

from pydantic import Field
from pydantic import NonNegativeFloat  # noqa:TC002

from gemseo.algos.opt.scipy_linprog.settings.base_scipy_linprog_settings import (
    BaseSciPyLinProgSettings,
)


class INTERIOR_POINT_Settings(BaseSciPyLinProgSettings):  # noqa: N801
    """The HiGHS interior point method setting."""

    _TARGET_CLASS_NAME = "INTERIOR_POINT"

    dual_feasibility_tolerance: NonNegativeFloat = Field(
        default=1e-7,
        description=("""The tolerance on the feasibility of the dual problem."""),
    )

    ipm_optimality_tolerance: NonNegativeFloat = Field(
        default=1e-8,
        ge=1e-12,
        description="""Optimality tolerance.""",
    )

    primal_feasibility_tolerance: NonNegativeFloat = Field(
        default=1e-7,
        description="""The tolerance on the feasibility of the primal problem.""",
    )
