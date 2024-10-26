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
"""Settings of the OAT DOE."""

from __future__ import annotations

from pydantic.types import PositiveFloat  # noqa: TCH002

from gemseo.algos.doe.base_doe_settings import BaseDOESettings
from gemseo.utils.pydantic_ndarray import NDArrayPydantic  # noqa: TCH001


class OATDOESettings(BaseDOESettings):
    """The settings of the OAT DOE."""

    initial_point: NDArrayPydantic
    """The initial point of the OAT DOE."""

    step: PositiveFloat = 0.05
    """The relative step of the OAT DOE so that the step in the ``x`` direction is
    ``step*(max_x-min_x)`` if ``x+step*(max_x-min_x)<=max_x`` and ``-step*(max_x-
    min_x)`` otherwise."""
