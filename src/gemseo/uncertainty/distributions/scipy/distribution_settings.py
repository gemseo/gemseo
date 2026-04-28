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

from typing import ClassVar

from pydantic import Field

from gemseo.uncertainty.distributions.base_settings import (
    BaseGenericDistributionSettings,
)
from gemseo.uncertainty.distributions.scipy.base_settings import (
    BaseSPDistributionSettings,
)


class SPDistribution_Settings(  # noqa: N801
    BaseGenericDistributionSettings, BaseSPDistributionSettings
):
    """The settings of an OpenTURNS-based distribution."""

    _LIBRARY_NAME: ClassVar[str] = "SciPy"

    interfaced_distribution: str = Field(
        default="uniform", description="The name of the probability distribution."
    )
