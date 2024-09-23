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

from typing import TYPE_CHECKING

from pydantic import Field

from gemseo.post.base_post_settings import BasePostSettings
from gemseo.utils.pydantic import update_field

if TYPE_CHECKING:
    from collections.abc import Sequence


class RadarChartSettings(BasePostSettings):  # noqa: D101
    iteration: int | None = Field(
        None,
        description="Either an iteration in :math:`-N,\\\\ldots,-1,1,`ldots,N` or "
        "``None`` for the iteration at which the optimum is located, where :math:`N` "
        "is the length of the database.",
    )
    constraint_names: Sequence[str] = Field(
        (),
        description="The names of the constraints. If empty, use all the "
        "constraints.",
    )
    show_names_radially: bool = Field(
        False,
        description="Whether to write the names of the constraints in the radial "
        "direction. Otherwise, write them horizontally. The radial "
        "direction can be useful for a high number of constraints.",
    )


update_field(RadarChartSettings, "fig_size", default=(6.4, 4.8))
