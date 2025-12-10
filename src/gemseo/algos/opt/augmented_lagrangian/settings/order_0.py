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
"""Settings for the augmented Lagrangian of order 0 algorithm."""

from __future__ import annotations

from pydantic import Field
from pydantic import PositiveFloat

from gemseo.algos.opt.augmented_lagrangian.settings.base import (  # noqa: E501
    BaseAugmentedLagrangianSettings,
)


class Augmented_Lagrangian_Order_0_Settings(BaseAugmentedLagrangianSettings):  # noqa: N801
    """The augmented Lagrangian of order 0 settings."""

    tau: PositiveFloat = Field(
        default=0.9,
        description="""The threshold to increase the penalty.""",
    )

    gamma: PositiveFloat = Field(
        default=1.5,
        description="""The penalty increase factor.""",
    )

    max_rho: PositiveFloat = Field(
        default=10_000,
        description="""The maximum penalty value.""",
    )


# TODO: API: remove
Augmented_Lagrangian_order_0_Settings = Augmented_Lagrangian_Order_0_Settings
