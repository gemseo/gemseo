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
"""Standard space cross-entropy importance sampling (SPCEIS) algorithm."""

from __future__ import annotations

from typing import ClassVar

from openturns import StandardSpaceCrossEntropyImportanceSampling

from gemseo.uncertainty.reliability.openturns.base_is import BaseOTImportanceSampling
from gemseo.uncertainty.reliability.openturns.is_spce_settings import (
    OT_IS_SPCE_Settings,
)


class OT_IS_SPCE(BaseOTImportanceSampling):  # noqa: N801
    """The standard space cross-entropy importance sampling (SPCEIS) algorithm."""

    settings_class: ClassVar[type[OT_IS_SPCE_Settings]] = OT_IS_SPCE_Settings

    _ALGO_CLASS: ClassVar[type[StandardSpaceCrossEntropyImportanceSampling]] = (
        StandardSpaceCrossEntropyImportanceSampling
    )
