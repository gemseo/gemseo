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
"""Settings for the central composite DOE from the pyDOE library."""

from __future__ import annotations

from pydantic import Field
from strenum import StrEnum

from gemseo.algos.doe.pydoe.settings.base_pydoe_settings import BasePyDOESettings


class Alpha(StrEnum):
    """A parameter to describe how the variance is distributed."""

    orthogonal = "orthogonal"
    o = "o"
    rotatable = "rotatable"
    r = "r"


class Face(StrEnum):
    """The relation between the start points and the corner (factorial) points."""

    circumscribed = "circumscribed"
    ccc = "ccc"
    inscribed = "inscribed"
    cci = "cci"
    faced = "faced"
    ccf = "ccf"


class PYDOE_CCDESIGN_Settings(BasePyDOESettings):  # noqa: N801
    """The settings for the central composite DOE from the pyDOE library."""

    _TARGET_CLASS_NAME = "PYDOE_CCDESIGN"

    alpha: Alpha = Field(
        default=Alpha.orthogonal,
        description="""A parameter to describe how the variance is distributed.

Either "orthogonal" or "rotatable".""",
    )

    center: tuple[int, int] = Field(
        default=(4, 4),
        description="The 2-tuple of center points for the central composite design.",
    )

    face: Face = Field(
        default=Face.circumscribed,
        description="The relation between the start and corner (factorial) points.",
    )
