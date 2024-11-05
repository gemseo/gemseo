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


class ParetoFront_Settings(BasePostSettings):  # noqa: D101, N801
    _TARGET_CLASS_NAME = "ParetoFront"
    show_non_feasible: bool = Field(
        default=True,
        description="Whether to show the non-feasible points in the plot.",
    )
    objectives: Sequence[str] = Field(
        default=(),
        description="The functions names or design variables to plot. If empty, "
        "use the objective function (maybe a vector).",
    )
    objectives_labels: Sequence[str] = Field(
        default=(),
        description="The labels of the objective components. If empty, use the "
        "objective name suffixed by an index.",
    )


update_field(ParetoFront_Settings, "fig_size", default=(10.0, 10.0))
