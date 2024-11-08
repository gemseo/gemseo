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
"""Settings for the SciPy DOEs."""

from __future__ import annotations

from enum import IntEnum

from pydantic import Field
from strenum import StrEnum

from gemseo.algos.doe.base_n_samples_based_doe_settings import (
    BaseNSamplesBasedDOESettings,
)


class Hypersphere(StrEnum):
    """The Hypersphere options."""

    volume = "volume"
    surface = "surface"


class Optimizer(StrEnum):
    """The optimization scheme to improve the quality of the DOE after sampling."""

    RANDOM_CD = "random-cd"
    LLOYD = "lloyd"


class Strength(IntEnum):
    """The strength of the LHS."""

    one = 1
    two = 2


class BaseSciPyDOESettings(BaseNSamplesBasedDOESettings):
    """The settings for the ``OpenTURNS`` DOE."""

    seed: int | None = Field(
        default=None,
        description="""The seed used for reproducibility reasons.

If ``None``, use :attr:`.seed`.""",
    )
