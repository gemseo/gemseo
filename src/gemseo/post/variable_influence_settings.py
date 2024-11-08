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

from pydantic import Field

from gemseo.post.base_post_settings import BasePostSettings


class VariableInfluence_Settings(BasePostSettings):  # noqa: D101, N801
    _TARGET_CLASS_NAME = "VariableInfluence"
    level: float = Field(
        default=0.99,
        description="The proportion of the total sensitivity to use as a threshold to "
        "filter the variables.",
        ge=0.0,
        le=1.0,
    )
    absolute_value: bool = Field(
        default=False,
        description="Whether to plot the absolute value of the influence.",
    )
    log_scale: bool = Field(
        default=False, description="Whether to set the y-axis as log scale."
    )
    save_var_files: bool = Field(
        default=False,
        description="Whether to save the influential variables indices to a NumPy "
        "file.",
    )
