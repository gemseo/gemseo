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
"""Settings for the SciPy-based joint probability distributions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from pydantic import Field

from gemseo.uncertainty.distributions.base_joint_settings import (
    BaseJointDistributionSettings,
)
from gemseo.utils.pydantic import BaseSettings


class SPJointDistribution_Settings(BaseJointDistributionSettings):  # noqa: N801
    """The settings of a SciPy-based joint probability distribution."""

    _LIBRARY_NAME: ClassVar[str] = "SciPy"

    marginal_settings: Sequence[BaseSettings] = Field(
        description="The SciPy-based marginal probability distributions."
    )
