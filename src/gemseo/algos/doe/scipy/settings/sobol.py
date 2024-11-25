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
"""Settings for the Sobol sequence DOE from the SciPy library."""

from __future__ import annotations

from pydantic import Field
from pydantic import PositiveInt  # noqa:TC002

from gemseo.algos.doe.scipy.settings.base_scipy_doe_settings import BaseSciPyDOESettings
from gemseo.algos.doe.scipy.settings.base_scipy_doe_settings import Optimizer


class Sobol_Settings(BaseSciPyDOESettings):  # noqa: N801
    """The settings for the Sobol sequence DOE from the SciPy library."""

    _TARGET_CLASS_NAME = "Sobol"

    optimization: Optimizer | None = Field(
        default=None,
        description="""The name of an optimization scheme to improve the DOE's quality.

If ``None``, use the DOE as is. New in SciPy 1.10.0.""",
    )

    scramble: bool = Field(
        default=True,
        description="""Whether to use scrambling (Owen type).

Only available with SciPy >= 1.10.0.""",
    )

    bits: PositiveInt = Field(
        30,
        le=64,
        description="""The number of bits of the generator.""",
    )
