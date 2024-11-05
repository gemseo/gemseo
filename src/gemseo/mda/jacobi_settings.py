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
"""Settings for MDAJacobi."""

from __future__ import annotations

from gemseo.algos.sequence_transformer.acceleration import AccelerationMethod
from gemseo.mda.base_parallel_mda_settings import BaseParallelMDASettings
from gemseo.utils.pydantic import copy_field


class MDAJacobi_Settings(BaseParallelMDASettings):  # noqa: N801
    """The settings for :class:`.MDAJacobi`."""

    acceleration_method: AccelerationMethod = copy_field(
        "acceleration_method",
        BaseParallelMDASettings,
        default=AccelerationMethod.ALTERNATE_2_DELTA,
    )
