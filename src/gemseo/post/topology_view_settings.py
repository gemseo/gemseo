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

from collections.abc import Sequence
from typing import Union

from pydantic import Field

from gemseo.post.base_post_settings import BasePostSettings
from gemseo.utils.pydantic import update_field


class TopologyViewSettings(BasePostSettings):  # noqa: D101
    n_x: int = Field(
        ...,
        description="The number of elements in the horizontal direction.",
        ge=1,
    )
    n_y: int = Field(
        ...,
        description="The number of elements in the vertical direction.",
        ge=1,
    )
    observable: str = Field(
        "",
        description="The name of the observable to be plotted. It should be of size "
        "``n_x*n_y``.",
    )
    iterations: Union[int, Sequence[int]] = Field(
        (),
        description="The iterations of the optimization history. If empty, "
        "the last iteration is taken.",
    )


update_field(TopologyViewSettings, "fig_size", default=(6.4, 4.8))
