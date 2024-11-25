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
"""Settings of the diagonal DOE for scalable model construction."""

from __future__ import annotations

from collections.abc import Sequence  # noqa: TC003

from pydantic import Field

from gemseo.algos.doe.base_n_samples_based_doe_settings import (
    BaseNSamplesBasedDOESettings,
)


class DiagonalDOE_Settings(BaseNSamplesBasedDOESettings):  # noqa: N801
    """The settings of the diagonal DOE for scalable model construction."""

    _TARGET_CLASS_NAME = "DiagonalDOE"

    n_samples: int = Field(
        2,
        ge=2,
        description="""The number of samples.

The number of samples must be greater than or equal than 2.""",
    )

    reverse: Sequence[str] = Field(
        default=(),
        description="""The dimensions or variables to sample from upper to lower bounds.

If empty, every dimension will be sampled from lower to upper bounds.""",
    )
