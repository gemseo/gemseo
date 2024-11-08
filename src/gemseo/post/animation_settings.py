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

from pathlib import Path
from typing import Any

from pydantic import ConfigDict
from pydantic import Field
from pydantic import NonNegativeInt
from pydantic import PositiveFloat
from pydantic import PositiveInt

from gemseo.post.base_post import BasePost
from gemseo.post.base_post_settings import BasePostSettings


class Animation_Settings(BasePostSettings):  # noqa: D101, N801
    # This is required to supporting the field post_processor.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    _TARGET_CLASS_NAME = "Animation"

    frame_rate: PositiveInt = Field(
        default=1,
        description="The number of iterations per time step.",
    )
    first_iteration: int = Field(
        default=-1, description="The iteration to begin the animation."
    )
    time_step: PositiveFloat = Field(
        default=100, description="The time step between two frames in milliseconds."
    )
    n_repetitions: NonNegativeInt = Field(
        default=0,
        description="The number of times the animation is played. "
        "If ``0``, play infinitely.",
    )
    temporary_database_path: str | Path = Field(
        default="",
        description="The path to a temporary database to avoid deepcopy memory errors."
        "If empty, deepcopy is used instead.",
    )
    gif_file_path: str | Path = Field(
        default="animated_gif",
        description="The path to the GIF file.",
    )
    remove_frames: bool = Field(
        default=True,
        description="Whether to remove the frame images after the GIF generation.",
    )
    post_processing: BasePost[Any] = Field(
        ...,
        description="The post processing object.",
    )
    post_processing_settings: BasePostSettings = Field(
        ...,
        description="The settings for the post processing object.",
    )
