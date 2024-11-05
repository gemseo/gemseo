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
from typing import TYPE_CHECKING

from pydantic import Field
from pydantic import model_validator

from gemseo.post.base_post_settings import BasePostSettings
from gemseo.utils.pydantic import update_field

if TYPE_CHECKING:
    from typing_extensions import Self


class OptHistoryView_Settings(BasePostSettings):  # noqa: D101, N801
    _TARGET_CLASS_NAME = "OptHistoryView"
    variable_names: Sequence[str] = Field(
        default=(),
        description="The names of the variables to display. "
        "If empty, use all design variables.",
    )
    obj_relative: bool = Field(
        default=False,
        description="Whether the difference between the objective and its initial "
        "value is plotted instead of the objective.",
    )
    obj_min: float | None = Field(
        default=None,
        description="The lower limit of the *y*-axis on which the objective is "
        "plotted. This limit must be less than or equal to the minimum "
        "value of the objective history. If ``None``, use the minimum "
        "value of the objective history.",
    )
    obj_max: float | None = Field(
        default=None,
        description="The upper limit of the *y*-axis on which the objective is "
        "plotted. This limit must be greater than or equal to the maximum "
        "value of the objective history. If ``None``, use the maximum "
        "value of the objective history.",
    )

    @model_validator(mode="after")
    def check_obj_min_max(self) -> Self:
        """Check that obj_min <= obj_max."""
        if (
            self.obj_min is not None
            and self.obj_max is not None
            and self.obj_min > self.obj_max
        ):
            msg = (
                f"The value of obj_min ({self.obj_min}) must be lower than the "
                f"value of obj_max ({self.obj_max})."
            )
            raise ValueError(msg)
        return self


update_field(OptHistoryView_Settings, "fig_size", default=(11.0, 6.0))
