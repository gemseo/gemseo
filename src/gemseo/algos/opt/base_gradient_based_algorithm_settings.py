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
"""Settings for gradient-based optimization algorithms."""

from __future__ import annotations

from numpy import inf
from pydantic import BaseModel
from pydantic import Field
from pydantic import NonNegativeFloat  # noqa: TCH002


class BaseGradientBasedAlgorithmSettings(BaseModel):
    """The settings for gradient-based optimization algorithms."""

    kkt_tol_abs: NonNegativeFloat = Field(
        default=inf,
        description=(
            """The absolute tolerance on the KKT residual norm.

            If ``inf`` this criterion is not activated.
            """
        ),
    )

    kkt_tol_rel: NonNegativeFloat = Field(
        default=inf,
        description=(
            """The relative tolerance on the KKT residual norm.

            If ``inf`` this criterion is not activated.
            """
        ),
    )
