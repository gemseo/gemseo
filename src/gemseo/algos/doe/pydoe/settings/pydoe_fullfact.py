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
"""Settings for the full factorial DOE from the PyDOE library."""

from __future__ import annotations

from collections.abc import Sequence  # noqa: TC003

from pydantic import Field
from pydantic import NonNegativeInt
from pydantic import PositiveInt  # noqa: TC002

from gemseo.algos.doe.pydoe.settings.base_pydoe_settings import BasePyDOESettings


class PYDOE_FULLFACT_Settings(BasePyDOESettings):  # noqa: N801
    """The settings for the full factorial DOE from the pyDOE library."""

    _TARGET_CLASS_NAME = "PYDOE_FULLFACT"

    n_samples: NonNegativeInt = Field(
        default=0,
        description="""The number of samples.

If 0, set from the settings.""",
    )

    levels: Sequence[PositiveInt] | PositiveInt = Field(
        default=(),
        description="""The levels.

One must either specify ``n_samples`` or ``levels``.  The levels are
inferred from the number of samples if the former is specified.""",
    )
