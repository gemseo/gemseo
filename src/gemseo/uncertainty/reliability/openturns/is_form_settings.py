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
"""The settings for the FORM-based importance sampling algorithm."""

from __future__ import annotations

from pydantic import Field

from gemseo.uncertainty.reliability.openturns.base_is_settings import BaseOTISSettings
from gemseo.uncertainty.reliability.openturns.form_settings import OT_FORM_Settings


class OT_IS_FORM_Settings(BaseOTISSettings):  # noqa: N801
    """The settings for the FORM-based importance sampling algorithm."""

    control: bool = Field(
        default=False,
        description="Whether the algorithm is controlled by the tangent hyperplan.",
    )

    form_settings: OT_FORM_Settings = Field(
        default_factory=OT_FORM_Settings,
        description="The settings of the FORM algorithm.",
    )
