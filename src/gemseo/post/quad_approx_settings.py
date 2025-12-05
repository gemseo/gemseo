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
from pydantic import NonNegativeInt

from gemseo.post.base_post_settings import BasePostSettings
from gemseo.typing import StrKeyMapping


class QuadApprox_Settings(BasePostSettings):  # noqa: D101, N801
    _INHERITED_FIELD_DEFAULTS: ClassVar[StrKeyMapping] = {"fig_size": (9.0, 6.0)}
    function: str = Field(
        ...,
        description="The function name to build the quadratic approximation.",
    )
    func_index: NonNegativeInt | None = Field(
        default=None,
        description="The index of the output of interest to be defined if the "
        "function has a multidimensional output. If `None` and if the "
        "output is multidimensional, an error is raised.",
    )
