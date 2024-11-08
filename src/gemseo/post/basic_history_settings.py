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


class BasicHistory_Settings(BasePostSettings):  # noqa: D101, N801
    _TARGET_CLASS_NAME = "BasicHistory"

    variable_names: Sequence[str] = Field(
        ...,
        description="The names of the variables.",
        min_length=1,
    )
    normalize: bool = Field(
        default=False,
        description="Whether to normalize the data.",
    )


update_field(BasicHistory_Settings, "fig_size", default=(11.0, 6.0))
