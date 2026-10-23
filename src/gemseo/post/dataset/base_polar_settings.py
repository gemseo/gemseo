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
"""Base settings for dataset visualizations using polar coordinates."""

from __future__ import annotations

from pydantic import Field

from gemseo.post.dataset.base_settings import BaseDatasetPlotSettings


class BasePolarDatasetPlotSettings(BaseDatasetPlotSettings):
    """The base settings for dataset visualizations using polar coordinates."""

    rmin: float | None = Field(
        default=None,
        description="The minimum value on the r-axis. If `None`, compute it from data.",
    )

    rmax: float | None = Field(
        default=None,
        description="The maximum value on the r-axis. If `None`, compute it from data.",
    )
