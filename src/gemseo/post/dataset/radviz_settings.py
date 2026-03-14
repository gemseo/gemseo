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

"""The settings for `RadViz`."""

from __future__ import annotations

from pydantic import Field

from gemseo.post.dataset.base_cartesian_settings import BaseCartesianDatasetPlotSettings


class RadViz_Settings(BaseCartesianDatasetPlotSettings):  # noqa: N801
    """The settings for `RadViz`."""

    classifier: str = Field(description="The name of the variable to group the data.")
