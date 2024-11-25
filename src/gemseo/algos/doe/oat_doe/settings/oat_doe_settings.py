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

from pydantic import Field
from pydantic.types import PositiveFloat  # noqa: TC002

from gemseo.algos.doe.base_doe_settings import BaseDOESettings
from gemseo.utils.pydantic_ndarray import NDArrayPydantic  # noqa: TC001


class OATDOE_Settings(BaseDOESettings):  # noqa: N801
    """The settings of the OAT DOE."""

    _TARGET_CLASS_NAME = "OATDOE"

    initial_point: NDArrayPydantic = Field(
        description="The initial point of the OAT DOE."
    )

    step: PositiveFloat = Field(
        default=0.05,
        description="""The relative step of the OAT DOE.

The step in the ``x`` direction is
step*(max_x-min_x)`` if ``x+step*(max_x-min_x)<=max_x`` and
``-step*(max_x- min_x)`` otherwise.""",
    )
