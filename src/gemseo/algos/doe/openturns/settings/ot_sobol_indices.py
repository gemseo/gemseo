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
"""Settings for the Sobol indices DOE from the OpenTURNS library."""

from __future__ import annotations

from pydantic import Field

from gemseo.algos.doe.openturns.settings.base_openturns_settings import (
    BaseOpenTURNSSettings,
)


class OT_SOBOL_INDICES_Settings(BaseOpenTURNSSettings):  # noqa: N801
    """The settings for the Sobol indices DOE from the OpenTURNS library."""

    _TARGET_CLASS_NAME = "OT_SOBOL_INDICES"

    eval_second_order: bool = Field(
        default=True,
        description="""Whether to build a DOE to evaluate also the second-order indices.

If ``False``, the DOE is designed for first and total-order indices only.""",
    )
