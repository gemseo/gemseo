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
"""Base settings for defining a triangular distribution."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field
from pydantic import model_validator

from gemseo.settings.base_settings import BaseSettings

if TYPE_CHECKING:
    from typing_extensions import Self


class BaseTriangularDistributionSettings(BaseSettings):  # noqa: N801
    """The base settings of a triangular distribution."""

    minimum: float = Field(
        default=0.0,
        description="The minimum of the triangular random variable.",
    )

    maximum: float = Field(
        default=1.0,
        description="The maximum of the triangular random variable.",
    )

    mode: float = Field(
        default=0.5,
        description="The mode of the triangular random variable.",
    )

    @model_validator(mode="after")
    def __validate(self) -> Self:
        if not (self.minimum < self.mode < self.maximum):
            msg = (
                "The parameters of the triangular distribution do not satisfy "
                "the order: minimum < mode < maximum."
            )
            raise ValueError(msg)

        return self
