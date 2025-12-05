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
from typing import ClassVar

from pydantic import Field

from gemseo.post.base_post_settings import BasePostSettings
from gemseo.typing import StrKeyMapping


class ScatterPlotMatrix_Settings(BasePostSettings):  # noqa: D101, N801
    _INHERITED_FIELD_DEFAULTS: ClassVar[StrKeyMapping] = {"fig_size": (10.0, 10.0)}
    filter_non_feasible: bool = Field(
        default=False,
        description="Whether to remove the non-feasible points from the data.",
    )
    variable_names: Sequence[str] = Field(
        default=(),
        description="The functions names or design variables to plot. If empty, "
        "plot all design variables.",
    )
