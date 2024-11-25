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

from typing import TYPE_CHECKING

from numpy import inf
from pydantic import BaseModel
from pydantic import Field
from pydantic import NonNegativeFloat  # noqa: TC002
from pydantic import model_validator

if TYPE_CHECKING:
    from typing_extensions import Self


class BaseGradientBasedAlgorithmSettings(BaseModel):
    """The settings for gradient-based optimization algorithms."""

    kkt_tol_abs: NonNegativeFloat = Field(
        default=inf,
        description="""The absolute tolerance on the KKT residual norm.

If ``inf`` this criterion is not activated.""",
    )

    kkt_tol_rel: NonNegativeFloat = Field(
        default=inf,
        description="""The relative tolerance on the KKT residual norm.

If ``inf`` this criterion is not activated.""",
    )

    @model_validator(mode="after")
    def __check_kkt_with_jacobian_in_database(self) -> Self:
        """Check the consistency of KKT options with Jacobian storage.

        Currently,
        KKT options can only be used along with ``store_jacobian=True``
        option (default option with all gradient-based algorithm).
        """
        if not self.store_jacobian and (
            self.kkt_tol_abs is not inf or self.kkt_tol_rel is not inf
        ):
            msg = "KKT options can only be set with store_jacobian=True"
            raise ValueError(msg)
        return self
