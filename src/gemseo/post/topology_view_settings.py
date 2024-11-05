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
from pydantic import PositiveInt

from gemseo.post.base_post_settings import BasePostSettings
from gemseo.utils.pydantic import update_field


class TopologyView_Settings(BasePostSettings):  # noqa: D101, N801
    _TARGET_CLASS_NAME = "TopologyView"
    n_x: PositiveInt = Field(
        ...,
        description="The number of elements in the horizontal direction.",
    )
    n_y: PositiveInt = Field(
        ...,
        description="The number of elements in the vertical direction.",
    )
    observable: str = Field(
        default="",
        description="The name of the observable to be plotted. It should be of size "
        "``n_x*n_y``.",
    )
    iterations: int | Sequence[int] = Field(
        default=(),
        description="The iterations of the optimization history. If empty, "
        "the last iteration is taken.",
    )


update_field(TopologyView_Settings, "fig_size", default=(6.4, 4.8))
