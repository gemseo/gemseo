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
"""Settings for the full factorial DOE from the OpenTURNS library."""

from __future__ import annotations

from collections.abc import Sequence  # noqa: TC003

from pydantic import Field
from pydantic import NonNegativeInt
from pydantic import PositiveInt  # noqa: TC002

from gemseo.algos.doe.openturns.settings.base_openturns_settings import (
    BaseOpenTURNSSettings,
)


class OT_FULLFACT_Settings(BaseOpenTURNSSettings):  # noqa: N801
    """The settings for the full factorial DOE from the OpenTURNS library."""

    _TARGET_CLASS_NAME = "OT_FULLFACT"

    n_samples: NonNegativeInt = Field(
        default=0,
        description="""The number of samples.

If 0, set from the options.""",
    )

    levels: PositiveInt | Sequence[PositiveInt] = Field(
        default=(),
        description="""The number of levels per input direction.

If scalar, this value is applied to each input direction.""",
    )
