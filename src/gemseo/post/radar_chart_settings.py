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
"""Settings for post-processing."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import Field

from gemseo.post.base_post_settings import BasePostSettings
from gemseo.utils.pydantic import update_field


class RadarChart_Settings(BasePostSettings):  # noqa: D101, N801
    _TARGET_CLASS_NAME = "RadarChart"
    iteration: int | None = Field(
        default=None,
        description=r"Either an iteration in :math:`\{-N,\ldots,-1,1,\ldots,N\}` or "
        "``None`` for the iteration at which the optimum is located, where :math:`N` "
        "is the length of the database.",
    )
    constraint_names: Sequence[str] = Field(
        default=(),
        description="The names of the constraints. If empty, use all the constraints.",
    )
    show_names_radially: bool = Field(
        default=False,
        description="Whether to write the names of the constraints in the radial "
        "direction. Otherwise, write them horizontally. The radial "
        "direction can be useful for a high number of constraints.",
    )


update_field(RadarChart_Settings, "fig_size", default=(6.4, 4.8))
