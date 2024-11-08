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

from gemseo.post.base_post_settings import BasePostSettings
from gemseo.utils.pydantic import update_field


class ParallelCoordinates_Settings(BasePostSettings):  # noqa: D101, N801
    _TARGET_CLASS_NAME = "ParallelCoordinates"


update_field(ParallelCoordinates_Settings, "fig_size", default=(10.0, 5.0))
