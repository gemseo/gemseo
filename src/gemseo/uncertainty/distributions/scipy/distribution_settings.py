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
"""Settings for the SciPy-based probability distributions."""

from __future__ import annotations

from collections.abc import Mapping  # noqa: TC003
from typing import Any
from typing import Final

from pydantic import Field

from gemseo import READ_ONLY_EMPTY_DICT
from gemseo.typing import StrKeyMapping  # noqa: TC001
from gemseo.uncertainty.distributions.base_distribution_settings import (
    BaseDistribution_Settings,
)

_INTERFACED_DISTRIBUTION: Final[str] = "uniform"
"""The default value of interfaced_distribution."""

_PARAMETERS: Final[READ_ONLY_EMPTY_DICT] = READ_ONLY_EMPTY_DICT
"""The default value of parameters."""

_STANDARD_PARAMETERS: Final[READ_ONLY_EMPTY_DICT] = READ_ONLY_EMPTY_DICT
"""The default value of standard_parameters."""


class SPDistribution_Settings(BaseDistribution_Settings):  # noqa: N801
    """The settings of an OpenTURNS-based distribution."""

    _TARGET_CLASS_NAME = "SPDistribution"

    interfaced_distribution: str = Field(
        default="uniform", description="The name of the probability distribution."
    )

    parameters: StrKeyMapping | tuple[Any, ...] = Field(
        default_factory=dict,
        description="The parameters of the probability distribution.",
    )

    standard_parameters: Mapping[str, str | int | float] = Field(
        default_factory=dict,
        description="""The parameters of the probability distribution
used for string representation only.""",
    )
