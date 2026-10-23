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
"""Settings for MDAGaussSeidelNewtonRaphson."""

from __future__ import annotations

from pydantic import Field

from gemseo.mda.gauss_seidel_settings import MDAGaussSeidel_Settings  # noqa: TC001
from gemseo.mda.newton_raphson_settings import MDANewtonRaphson_Settings  # noqa: TC001
from gemseo.mda.sequential_settings import MDASequential_Settings


class MDAGaussSeidelNewtonRaphson_Settings(MDASequential_Settings):  # noqa: N801
    """The settings for [MDAGaussSeidelNewtonRaphson][gemseo.mda.gauss_seidel_newton_raphson.MDAGaussSeidelNewtonRaphson]."""  # noqa: E501

    gauss_seidel_settings: MDAGaussSeidel_Settings = Field(
        default_factory=MDAGaussSeidel_Settings,
        description="The settings for the Gauss-Seidel MDA.",
    )

    newton_raphson_settings: MDANewtonRaphson_Settings = Field(
        default_factory=MDANewtonRaphson_Settings,
        description="The settings for the Newton-Raphson MDA.",
    )
