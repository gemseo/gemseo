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
"""Settings for the LHS DOE from the pyDOE library."""

from __future__ import annotations

from numpy.random import Generator
from pydantic import Field
from pydantic import NonNegativeInt
from pydantic import PositiveInt  # noqa: TC002
from strenum import StrEnum

from gemseo.doe.pydoe.settings.base_pydoe_settings import BasePyDOESettings


class Criterion(StrEnum):
    """The criteria for the LHS."""

    CENTER = "center"
    C = "c"
    MAXIMIN = "maximin"
    M = "m"
    CENTERMAXIMIN = "centermaximin"
    CM = "cm"
    CORRELATION = "correlation"
    CORR = "corr"
    LHSMU = "lhsmu"


class PYDOE_LHS_Settings(BasePyDOESettings):  # noqa: N801
    """The settings for the LHS DOE from the pyDOE library."""

    criterion: Criterion | None = Field(
        default=None,
        description="""The criterion to use when sampling the points.

If `None`, randomize the points within the intervals.""",
    )

    iterations: PositiveInt = Field(
        default=5,
        description="The number of iterations in the `correlation`/`maximin` algorithms.",  # noqa: E501
    )

    n_samples: PositiveInt = Field(default=1, description="The number of samples.")

    seed: NonNegativeInt | Generator | None = Field(
        default=None,
        description="""Seed or NumPy random `Generator` which controls random draws.

If `None`, use
[BaseDOELibrary.seed][gemseo.doe.core.base_doe_library.BaseDOELibrary.seed].""",
    )
