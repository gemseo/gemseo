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


class Correlations_Settings(BasePostSettings):  # noqa: D101, N801
    _TARGET_CLASS_NAME = "Correlations"
    n_plots_x: PositiveInt = Field(
        default=5,
        description="The number of horizontal plots.",
    )
    n_plots_y: PositiveInt = Field(
        default=5,
        description="The number of vertical plots.",
    )
    coeff_limit: float = Field(
        default=0.95,
        description="The minimum correlation coefficient below which the variable "
        "is not plotted.",
        ge=0.0,
        le=1.0,
    )
    func_names: Sequence[str] = Field(
        default=(),
        description="The names of the functions to be considered. "
        "If empty, all the functions are considered.",
    )


update_field(Correlations_Settings, "fig_size", default=(15.0, 10.0))
