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
"""Settings for probability distributions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from typing import ClassVar

from pydantic import Field

from gemseo.algos.base_settings import BaseSettings
from gemseo.typing import StrKeyMapping


class BaseDistributionSettings(BaseSettings):  # noqa: N801
    """The base class for the settings of a probability distribution."""

    _LIBRARY_NAME: ClassVar[str]
    """The name of the library implementing the probability distribution."""


class BaseGenericDistributionSettings(BaseDistributionSettings):  # noqa: N801
    """The base class for the settings of a generic probability distribution."""

    interfaced_distribution: str = Field(
        description="The name of the probability distribution."
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
