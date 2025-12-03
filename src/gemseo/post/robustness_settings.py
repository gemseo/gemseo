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

from typing import ClassVar

from pydantic import Field

from gemseo.post.base_post_settings import BasePostSettings
from gemseo.typing import StrKeyMapping


class Robustness_Settings(BasePostSettings):  # noqa: D101, N801
    _FIELD_DEFAULTS: ClassVar[StrKeyMapping] = {"fig_size": (8.0, 5.0)}
    stddev: float = Field(
        default=0.01,
        description="The standard deviation of the normal uncertain variable to be "
        "added to the optimal design value; expressed as a fraction of "
        "the bounds of the design variables.",
        ge=0.0,
        le=1.0,
    )
